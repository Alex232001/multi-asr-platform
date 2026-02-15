from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import numpy as np
import io
import json
import os
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import wave
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vosk Transcriber 
class VoskTranscriber:
    def __init__(self):
        self.model = None

    def load_model(self):
        logger.info("Загрузка модели VOSK...")
        model_path = "ai_model/vosk-model-small-ru-0.22"
        if not os.path.exists(model_path):
            raise Exception(f"Модель не найдена: {model_path}")
        self.model = Model(model_path)
        logger.info("Модель VOSK загружена успешно")

    def transcribe(self, wf) -> dict:
        start_time = time.time()
        
        rec = KaldiRecognizer(self.model, wf.getframerate())
        rec.SetWords(True)
        results = []

        while True:
            data = wf.readframes(4096)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if 'result' in result:
                    for word in result['result']:
                        results.append({
                            'word': word['word'],
                            'confidence': word['conf']
                        })

        final_result = json.loads(rec.FinalResult())
        if 'result' in final_result:
            for word in final_result['result']:
                results.append({
                    'word': word['word'],
                    'confidence': word['conf']
                })

        text = " ".join(w['word'] for w in results)
        confidence = np.mean([w['confidence'] for w in results]) if results else 0.0
        processing_time = time.time() - start_time
        
        logger.info(f"Распознано {len(results)} слов, время: {processing_time:.2f} сек")
        
        return {
            "text": text,
            "confidence":  float(round(confidence, 4)),  # 0.95 вместо 95%
            "model": "vosk-ru",
            "words_count": len(results)
        }

# Конвертация аудио в WAV 
def convert_audio_to_wav(audio_data: bytes, content_type: str) -> bytes:
    try:
        # Определяем формат по Content-Type
        file_format = None
        
        if content_type:
            # Извлекаем формат из content-type
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
            else:
                # Пробуем определить по расширению в content-type
                parts = content_type.split('/')
                if len(parts) > 1 and parts[1]:
                    file_format = parts[1].lower()
        
        logger.info(f"Конвертация аудио, content-type: {content_type}, формат: {file_format}")
        
        # Если формат не определен, пробуем автоматическое определение
        if not file_format:
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_data))
                audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            except:
                # Пробуем как WAV
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

# Инициализация модели
transcriber = VoskTranscriber()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запуск при старте
    logger.info("Запуск VOSK сервиса...")
    try:
        transcriber.load_model()
        logger.info("VOSK сервис готов к работе")
    except Exception as e:
        logger.error(f"Ошибка инициализации VOSK: {str(e)}")
        raise
    yield
    # Очистка при завершении
    logger.info("Завершение VOSK сервиса...")

# Создание приложения с lifespan
app = FastAPI(title="Vosk Speech Recognition", lifespan=lifespan)

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
    logger.info(f"Получен запрос на распознавание")

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
                # Пробуем обработать как есть, может быть уже WAV
                pass

        # Обрабатываем через Vosk
        try:
            audio_buffer = io.BytesIO(audio_data)
            with wave.open(audio_buffer, "rb") as wf:
                result = transcriber.transcribe(wf)
                total_time = time.time() - start_time
                
                result["processing_time"] = round(total_time, 2)
                result["file_size_kb"] = round(len(audio_data) / 1024, 1)
                
                logger.info(f"Успешно обработано за {result['processing_time']} сек")
                return result
        except wave.Error as e:
            logger.error(f"Ошибка чтения WAV: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Некорректный WAV файл: {str(e)}")
        except Exception as e:
            logger.error(f"Ошибка обработки VOSK: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

# Обработчик для обратной совместимости с multipart 
@app.post("/transcribe-multipart")
async def transcribe_multipart(file: UploadFile = File(...)):
    """Совместимость со старым форматом"""
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
    status = "healthy" if transcriber.model else "unhealthy"
    return {
        "status": status, 
        "model": "vosk", 
        "ready": transcriber.model is not None,
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    return {
        "message": "Vosk Speech Recognition API", 
        "status": "running",
        "endpoints": [
            "POST /transcribe (raw audio)",
            "POST /transcribe-multipart (form-data)",
            "GET /health"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
