from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import numpy as np
import io
import json
import os
import librosa
import sentencepiece as spm
import onnxruntime as ort
from pydub import AudioSegment
import tempfile
import logging
import wave

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """Препроцессор идентичный NeMo"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400  # 25ms
        self.hop_length = 160  # 10ms
        self.n_mels = 80
        self.window = 'hann'
        self.f_min = 0
        self.f_max = 8000
        self.dither = 1e-05
        self.preemph = 0.97
        self.log_zero_guard_value = 2**-24
        
    def compute_mel_spectrogram(self, audio):
        """Вычисление Mel-спектрограммы"""
        # Пре-эмфаза
        audio = np.append(audio[0], audio[1:] - self.preemph * audio[:-1])
        
        # STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True
        )
        
        # Амплитудный спектр
        magnitude = np.abs(stft)
        
        # Mel-фильтры
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            norm='slaney'
        )
        
        # Применение Mel-фильтров
        mel_spectrogram = np.dot(mel_basis, magnitude)
        
        # Логарифмирование
        log_mel = np.log(np.clip(mel_spectrogram, a_min=self.log_zero_guard_value, a_max=None))
        
        return log_mel
    
    def normalize_batch(self, features, seq_len):
        """Нормализация признаков"""
        mean = features.mean(axis=2, keepdims=True)
        std = features.std(axis=2, keepdims=True)
        normalized = (features - mean) / (std + 1e-5)
        return normalized, seq_len
    
    def __call__(self, audio_signal, audio_length):
        """Основной метод препроцессинга"""
        batch_size = audio_signal.shape[0]
        features_list = []
        features_lengths = []
        
        for i in range(batch_size):
            audio = audio_signal[i]
            length = audio_length[i]
            audio = audio[:length]
            
            mel_spec = self.compute_mel_spectrogram(audio)
            features_list.append(mel_spec)
            features_lengths.append(mel_spec.shape[1])
        
        # Собираем батч
        max_length = max(features_lengths)
        batch_features = np.zeros((batch_size, self.n_mels, max_length), dtype=np.float32)
        
        for i, feat in enumerate(features_list):
            batch_features[i, :, :feat.shape[1]] = feat
        
        features_lengths = np.array(features_lengths, dtype=np.int64)
        
        # Нормализация
        batch_features, features_lengths = self.normalize_batch(batch_features, features_lengths)
        
        return batch_features, features_lengths


def decode_with_sentencepiece(sp, tokens):
    """Декодирование токенов с помощью SentencePiece"""
    try:
        valid_tokens = [t for t in tokens if t < sp.vocab_size()]
        
        if not valid_tokens:
            return ""
        
        try:
            text = sp.decode(valid_tokens)
            return text
        except:
            pieces = []
            for token_id in valid_tokens:
                try:
                    piece = sp.id_to_piece(int(token_id))
                    pieces.append(piece)
                except:
                    continue
            
            if pieces:
                text = sp.decode_pieces(pieces)
                return text
            else:
                return ""
                
    except Exception as e:
        logger.error(f"Ошибка при декодировании: {e}")
        return ""

def greedy_batch_decode_with_probs(logprobs, lengths):
    """Жадное декодирование батча с возвратом вероятностей"""
    batch_size = logprobs.shape[0]
    predictions = []
    probabilities = []
    
    for i in range(batch_size):
        seq_len = lengths[i]
        if seq_len > logprobs.shape[1]:
            seq_len = logprobs.shape[1]
            
        seq_logprobs = logprobs[i, :seq_len]
        best_tokens = np.argmax(seq_logprobs, axis=1)
        best_probs = np.exp(np.max(seq_logprobs, axis=1))
        
        decoded_tokens = []
        decoded_probs = []
        prev_token = -1
        
        for token_idx, prob in zip(best_tokens, best_probs):
            if token_idx != prev_token:
                if token_idx != 128 and token_idx < 128:
                    decoded_tokens.append(int(token_idx))
                    decoded_probs.append(float(prob))
                prev_token = token_idx
        
        predictions.append(decoded_tokens)
        probabilities.append(decoded_probs)
    
    return predictions, probabilities

def convert_numpy_types(obj):
    """Рекурсивно преобразует NumPy типы в стандартные Python типы"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# Конвертация аудио 
def convert_audio_to_wav(audio_data: bytes, content_type: str = "") -> bytes:
    """Конвертация аудио в WAV формат"""
    try:
        file_format = None
        
        if content_type:
            if 'wav' in content_type:
                file_format = 'wav'
            elif 'mp3' in content_type or 'mpeg' in content_type:
                file_format = 'mp3'
            elif 'ogg' in content_type:
                file_format = 'ogg'
            elif 'flac' in content_type:
                file_format = 'flac'
            elif 'm4a' in content_type or 'aac' in content_type:
                file_format = 'm4a'
        
        logger.info(f"Конвертация аудио, content-type: {content_type}, формат: {file_format}")
        
        if not file_format:
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_data))
                audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            except:
                file_format = 'wav'
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format='wav')
                audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        else:
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=file_format)
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        
        output = io.BytesIO()
        audio.export(output, format="wav")
        return output.getvalue()
    except Exception as e:
        logger.error(f"Ошибка конвертации: {str(e)}")
        raise Exception(f"Ошибка конвертации аудио: {str(e)}")

# ONNX Transcriber 
class ONNXTranscriber:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.preprocessor = None

    def load_model(self):
        logger.info("Загрузка ONNX модели...")
        try:
            model_path = "model.onnx"
            tokenizer_path = "tokenizer.model"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX модель не найдена: {model_path}")
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Токенизатор не найден: {tokenizer_path}")
            
            self.model = ort.InferenceSession(model_path)
            self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
            self.preprocessor = AudioPreprocessor()
            
            logger.info(f"✅ ONNX модель загружена. Размер словаря: {self.tokenizer.vocab_size()}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise

    def transcribe_audio_bytes(self, audio_data: bytes) -> dict:
        """Транскрибация из байтов аудио"""
        start_time = time.time()
        
        if not self.model or not self.tokenizer:
            raise Exception("Модель или токенизатор не загружены")
        
        try:
            # Сохраняем временный файл для обработки
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            # Загружаем и обрабатываем аудио
            audio, sr = librosa.load(temp_file_path, sr=16000)
            
            # Препроцессинг аудио
            audio_array = np.expand_dims(audio, axis=0).astype(np.float32)
            audio_length = np.array([audio_array.shape[1]], dtype=np.int64)
            
            processed_audio, processed_audio_length = self.preprocessor(audio_array, audio_length)
            
            # Распознавание с ONNX моделью
            onnx_inputs = {
                'audio_signal': processed_audio.astype(np.float32),
                'length': processed_audio_length.astype(np.int64)
            }
            
            logprobs = self.model.run(None, onnx_inputs)[0]
            
            # Декодируем результат
            token_sequences, probability_sequences = greedy_batch_decode_with_probs(
                logprobs, processed_audio_length
            )
            
            # Конвертируем токены в текст
            if token_sequences[0]:
                text_result = decode_with_sentencepiece(self.tokenizer, token_sequences[0])
                avg_confidence = np.mean(probability_sequences[0]) if probability_sequences[0] else 0.0
            else:
                text_result = ""
                avg_confidence = 0.0

            end_time = time.time()
            execution_time = end_time - start_time

            # Создаем результат
            result = {
                'text': text_result,
                'confidence': float(round(avg_confidence, 4)),
                'processing_time': float(round(execution_time, 2)),
                'model': 'onnx-nemo-ru',
                'tokens_count': len(token_sequences[0]) if token_sequences[0] else 0
            }
            
            return convert_numpy_types(result)
        
        except Exception as e:
            logger.error(f"Ошибка транскрибации: {e}")
            raise
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

# Инициализация
transcriber = ONNXTranscriber()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запуск при старте
    logger.info("Запуск ONNX Nemo сервиса...")
    try:
        transcriber.load_model()
        logger.info("ONNX Nemo сервис готов к работе")
    except Exception as e:
        logger.error(f"Ошибка инициализации ONNX: {str(e)}")
        raise
    yield
    # Очистка при завершении
    logger.info("Завершение ONNX Nemo сервиса...")

# Создание приложения с lifespan
app = FastAPI(title="ONNX Nemo Speech Recognition", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Основной обработчик: принимает сырое аудио 
@app.post("/transcribe")
async def transcribe_audio(request: Request):
    start_time = time.time()
    logger.info(f"Получен RAW запрос на распознавание")

    try:
        # Читаем сырое тело
        audio_data = await request.body()
        logger.info(f"Размер аудио данных: {len(audio_data)} bytes")
        
        if not audio_data or len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Пустое тело запроса")

        content_type = request.headers.get("content-type", "").lower()
        logger.info(f"Content-Type: {content_type}")

        # Конвертируем, если не WAV
        if not content_type.startswith("audio/wav"):
            try:
                logger.info("Конвертация аудио в WAV...")
                audio_data = convert_audio_to_wav(audio_data, content_type)
                logger.info(f"После конвертации: {len(audio_data)} bytes")
            except Exception as e:
                logger.error(f"Ошибка конвертации: {str(e)}")
                # Пробуем обработать как есть
                pass

        # Обрабатываем через ONNX модель
        try:
            result = transcriber.transcribe_audio_bytes(audio_data)
            total_time = time.time() - start_time
            
            result["file_size_kb"] = round(len(audio_data) / 1024, 1)
            result["total_processing_time"] = round(total_time, 2)
            
            logger.info(f"Успешно обработано за {result['total_processing_time']} сек")
            return result
        except Exception as e:
            logger.error(f"Ошибка обработки ONNX: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

# Обработчик для обратной совместимости с multipart 
@app.post("/transcribe-multipart")
async def transcribe_multipart(file: UploadFile = File(...)):
    """Совместимость со старым форматом multipart/form-data"""
    logger.info(f"Multipart запрос: {file.filename}, {file.content_type}")
    
    try:
        audio_data = await file.read()
        content_type = file.content_type or f"audio/{file.filename.split('.')[-1] if '.' in file.filename else 'wav'}"
        
        # Оборачиваем в Request для единой обработки
        class MockRequest:
            def __init__(self, data, content_type):
                self.body_data = data
                self.content_type = content_type
            
            async def body(self):
                return self.body_data
            
            def headers(self):
                return {"content-type": self.content_type}
        
        mock_request = MockRequest(audio_data, content_type)
        return await transcribe_audio(mock_request)
        
    except Exception as e:
        logger.error(f"Ошибка multipart: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    status = "healthy" if transcriber.model and transcriber.tokenizer else "unhealthy"
    return {
        "status": status, 
        "model": "onnx-nemo", 
        "ready": transcriber.model is not None and transcriber.tokenizer is not None,
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    return {
        "message": "ONNX Nemo Speech Recognition API", 
        "status": "running",
        "model": "onnx-nemo-ru",
        "endpoints": [
            "POST /transcribe (raw audio)",
            "POST /transcribe-multipart (form-data)",
            "GET /health"
        ],
        "supported_formats": ["wav", "mp3", "ogg", "flac", "m4a"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
