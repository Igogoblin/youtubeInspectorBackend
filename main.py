import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI

app = FastAPI(title="YouTube Bias Detector API (Groq)")

# Разрешаем запросы из браузерного расширения (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Укажите ваш API-ключ Groq здесь или передайте через переменную окружения GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Инициализируем клиент OpenAI, настроенный на серверы Groq
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)


class AnalyzeRequest(BaseModel):
    video_id: str


def extract_text_from_data(data) -> str:
    """Универсальное извлечение текста из объектов субтитров"""
    texts = []
    for item in data:
        if isinstance(item, dict):
            texts.append(item.get('text', ''))
        elif hasattr(item, 'text'):
            texts.append(item.text)
        elif hasattr(item, '__getitem__'):
            try:
                texts.append(item['text'])
            except Exception:
                texts.append(str(item))
        else:
            texts.append(str(item))
    return " ".join(texts).strip()


def get_youtube_transcript(video_id: str) -> str:
    """Универсальный и безопасный забор субтитров"""
    try:
        data = YouTubeTranscriptApi.get_transcript(video_id, languages=['ru', 'ru-RU', 'en', 'en-US'])
        text = extract_text_from_data(data)
        if text:
            return text
    except Exception:
        pass

    try:
        if hasattr(YouTubeTranscriptApi, 'list_transcripts'):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        else:
            transcript_list = YouTubeTranscriptApi().list(video_id)

        try:
            transcript = transcript_list.find_transcript(['ru', 'en'])
        except Exception:
            first_available = list(transcript_list)[0]
            transcript = first_available.translate('ru')

        data = transcript.fetch()
        text = extract_text_from_data(data)
        if text:
            return text

    except Exception as e:
        print(f"Ошибка получения субтитров: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Не удалось извлечь субтитры из данного видео: {str(e)}"
        )

    raise HTTPException(status_code=400, detail="Субтитры пустые или недоступны.")


@app.post("/api/analyze")
async def analyze_video(request: AnalyzeRequest):
    video_id = request.video_id
    
    # 1. Получаем субтитры
    transcript_text = get_youtube_transcript(video_id)
    
    # Ограничиваем длинный текст (~15 000 символов)
    max_chars = 15000
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[:max_chars] + "..."

    # 2. Промпт для ИИ
    system_prompt = """
    Ты — эксперт по когнитивной психологии, критическому мышлению и логике.
    Твоя задача — проанализировать предоставленный транскрипт видео и найти в нём когнитивные ошибки (искажения), манипуляции или логические неувязки.

    Отвечай строго в формате JSON со следующей структурой:
    {
      "has_biases": true,
      "summary": "Краткое резюме об общем качестве аргументации в видео (1-2 предложения).",
      "biases": [
        {
          "name": "Название когнитивной ошибки/манипуляции (на русском)",
          "quote": "Примерная цитата или тезис из видео",
          "explanation": "Объяснение, почему это является ошибкой мышления в данном контексте"
        }
      ]
    }

    Если когнитивных ошибок не найдено, укажи "has_biases": false и верни пустой массив "biases".
    Ищи такие ошибки как: Подтверждение предвзятости (Confirmation Bias), Подмена понятия (Straw Man), Апелляция к авторитету/эмоциям, Ошибка выжившего, Черно-белое мышление и т.д.
    """

    try:
        # Используем модель llama-3.3-70b-versatile от Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Транскрипт видео:\n{transcript_text}"}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка взаимодействия с Groq AI: {str(e)}")


@app.get("/")
def read_root():
    return {"status": "Server is running (Groq Backend)"}