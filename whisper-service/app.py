from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import subprocess
import json
import numpy as np
import time
import re
import os
from pydub import AudioSegment
import io
import tempfile
import logging
import wave

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Вспомогательные функции для Whisper 
def calculate_confidence_from_segments(segments):
    """Рассчитываем уверенность на основе сегментов транскрипции"""
    if not segments:
        return 0.5
    
    total_confidence = 0
    segment_count = 0
    
    for segment in segments:
        text = segment.get('text', '').strip()
        if text:
            segment_confidence = min(0.8, 0.3 + (len(text) * 0.02))
            total_confidence += segment_confidence
            segment_count += 1
    
    if segment_count > 0:
        return round(total_confidence / segment_count, 4)
    else:
        return 0.5

def extract_word_details_from_segments(segments):
    """Извлекаем детали о словах из сегментов"""
    word_details = []
    
    if not segments:
        return word_details
    
    for segment in segments:
        text = segment.get('text', '').strip()
        if text:
            words = re.findall(r'\b\w+\b', text)
            for word in words:
                word_confidence = 0.7
                word_details.append({
                    'word': word,
                    'confidence': word_confidence,
                    'confidence_level': get_confidence_level(word_confidence)
                })
    
    return word_details

def get_confidence_level(confidence):
    """Определяем уровень уверенности по числовому значению"""
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.6:
        return "medium"
    else:
        return "low"

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

class WhisperCLITranscriber:
    def __init__(self):
        self.model_path = "ggml_small.bin"
        self.model_loaded = False

    def load_model(self):
        """Загрузка и проверка модели"""
        logger.info("Инициализация Whisper-CLI...")
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Модель {self.model_path} не найдена")
            
            # Проверяем доступность whisper-cli
            try:
                result = subprocess.run(
                    ["whisper-cli", "--help"], 
                    capture_output=True, 
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info("Whisper-CLI доступен")
                    self.model_loaded = True
                else:
                    logger.warning(f"Whisper-CLI возвращает ошибку: {result.stderr}")
            except FileNotFoundError:
                logger.error("whisper-cli не найден в PATH")
            except Exception as e:
                logger.error(f"Ошибка при проверке whisper-cli: {e}")
                
        except Exception as e:
            logger.error(f"Ошибка инициализации Whisper-CLI: {e}")
            raise
    
    def transcribe_audio_bytes(self, audio_data: bytes) -> dict:
        """Транскрибация из байтов аудио"""
        start_time = time.time()
        
        if not self.model_loaded:
            raise Exception("Whisper-CLI не инициализирован")
        
        temp_audio_path = None
        temp_json_path = None
        
        try:
            # Создаем временный файл для аудио
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name

            # Создаем временный файл для JSON вывода
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_json:
                temp_json_path = temp_json.name

            logger.info(f"Запуск whisper-cli с файлом {temp_audio_path}")
            
            # Команда для whisper-cli
            cmd = [
                "whisper-cli", 
                "-f", temp_audio_path,
                "-m", self.model_path,
                "--output-json",
                "--output-file", temp_json_path.replace('.json', ''),
                "--language", "ru",
                "-pp"  # Параллельная обработка
            ]
            
            logger.info(f"Выполнение команды: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                errors='ignore',
                timeout=300  # 5 минут таймаут
            )
            
            if result.returncode != 0:
                logger.error(f"whisper-cli ошибка: {result.stderr}")
                raise Exception(f"whisper-cli ошибка: {result.stderr[:200]}")
            
            logger.info("whisper-cli успешно завершил работу")
            
            # Читаем JSON результат
            json_data = None
            if os.path.exists(temp_json_path):
                with open(temp_json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            
            if not json_data:
                raise Exception("Не удалось получить JSON результат от whisper-cli")
            
            # Формируем полный текст из сегментов
            full_text = ""
            segments = json_data.get('transcription', [])
            for segment in segments:
                segment_text = segment.get('text', '').strip()
                if segment_text:
                    full_text += segment_text + " "
            
            full_text = full_text.strip()
            
            # Рассчитываем уверенность
            overall_confidence = calculate_confidence_from_segments(segments)
            
            # Получаем детали по словам
            word_details = extract_word_details_from_segments(segments)
            
            # Формируем breakdown уверенности
            confidence_breakdown = {
                "overall_confidence": overall_confidence,
                "word_details": word_details,
                "high_confidence_count": len([w for w in word_details if w['confidence'] >= 0.8]),
                "medium_confidence_count": len([w for w in word_details if 0.5 <= w['confidence'] < 0.8]),
                "low_confidence_count": len([w for w in word_details if w['confidence'] < 0.5]),
                "total_words_analyzed": len(word_details)
            }
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Формируем результат
            result_data = {
                'text': full_text,
                'confidence': float(overall_confidence),
                'processing_time': float(round(execution_time, 2)),
                'model': 'whisper-cli-ru',
                'language': json_data.get('result', {}).get('language', 'ru'),
                'confidence_breakdown': confidence_breakdown,
                'segments_count': len(segments),
                'segments': segments[:10]  # Ограничиваем количество сегментов в ответе
            }
            
            logger.info(f"Распознано: {len(full_text)} символов, уверенность: {overall_confidence:.2f}")
            
            return convert_numpy_types(result_data)
        
        except subprocess.TimeoutExpired:
            logger.error("Таймаут выполнения whisper-cli (5 минут)")
            raise Exception("Таймаут обработки аудио")
        except Exception as e:
            logger.error(f"Ошибка транскрибации: {e}")
            raise
        finally:
            # Очистка временных файлов
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
            if temp_json_path and os.path.exists(temp_json_path):
                try:
                    os.unlink(temp_json_path)
                except:
                    pass

# Инициализация модели
transcriber = WhisperCLITranscriber()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запуск при старте
    logger.info("Запуск Whisper-CLI сервиса...")
    try:
        transcriber.load_model()
        logger.info("Whisper-CLI сервис готов к работе")
    except Exception as e:
        logger.error(f"Ошибка инициализации Whisper-CLI: {str(e)}")
        # Не падаем, чтобы сервис мог работать в демо-режиме
    yield
    # Очистка при завершении
    logger.info("Завершение Whisper-CLI сервиса...")

# Создание приложения с lifespan
app = FastAPI(title="Whisper-CLI Speech Recognition", lifespan=lifespan)

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
    logger.info("Получен RAW запрос на распознавание")

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

        # Проверяем, что это валидный WAV
        try:
            audio_buffer = io.BytesIO(audio_data)
            with wave.open(audio_buffer, "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                logger.info(f"WAV параметры: {sample_rate}Hz, {channels} каналов")
        except wave.Error as e:
            logger.error(f"Некорректный WAV файл: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Некорректный аудиофайл: {str(e)}")

        # Обрабатываем через Whisper-CLI
        try:
            result = transcriber.transcribe_audio_bytes(audio_data)
            total_time = time.time() - start_time
            
            result["file_size_kb"] = round(len(audio_data) / 1024, 1)
            result["total_processing_time"] = round(total_time, 2)
            
            logger.info(f"Успешно обработано за {result['total_processing_time']} сек")
            return result
        except Exception as e:
            logger.error(f"Ошибка обработки Whisper-CLI: {str(e)}")
            
            # Демо-результат если whisper-cli недоступен
            if not transcriber.model_loaded:
                logger.info("Используем демо-результат")
                demo_text = "Это демонстрационный результат от OpenAI Whisper. Модель показывает высокую точность распознавания речи на русском языке."
                
                return {
                    "text": demo_text,
                    "confidence": 0.85,
                    "processing_time": round(total_time, 2),
                    "model": "whisper-demo",
                    "language": "ru",
                    "file_size_kb": round(len(audio_data) / 1024, 1),
                    "total_processing_time": round(total_time, 2),
                    "note": "Демо-режим (whisper-cli недоступен)"
                }
            
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
    status = "healthy" if transcriber.model_loaded else "demo_mode"
    return {
        "status": status, 
        "model": "whisper-cli", 
        "ready": transcriber.model_loaded,
        "timestamp": time.time(),
        "mode": "demo" if not transcriber.model_loaded else "production"
    }

@app.get("/")
async def root():
    return {
        "message": "Whisper-CLI Speech Recognition API", 
        "status": "running",
        "model": "whisper-small-ru",
        "endpoints": [
            "POST /transcribe (raw audio)",
            "POST /transcribe-multipart (form-data)",
            "GET /health"
        ],
        "supported_formats": ["wav", "mp3", "ogg", "flac", "m4a"],
        "note": "Использует whisper-cli с ggml моделью"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
