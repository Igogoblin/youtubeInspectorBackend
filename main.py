import os
import json
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)


class AnalyzeRequest(BaseModel):
    video_id: str


def get_youtube_transcript_raw(video_id: str):
    """Безопасный забор субтитров с сохранением таймкодов"""
    try:
        data = YouTubeTranscriptApi.get_transcript(video_id, languages=['ru', 'ru-RU', 'en', 'en-US'])
        if data:
            return data
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
        if data:
            return data

    except Exception as e:
        print(f"Ошибка получения субтитров: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Не удалось извлечь субтитры из данного видео: {str(e)}"
        )

    raise HTTPException(status_code=400, detail="Субтитры пустые или недоступны.")


def format_transcript_with_timestamps(data) -> str:
    """Форматирует субтитры с указанием секунд для ИИ"""
    formatted_lines = []
    total_length = 0
    max_chars = 15000

    for item in data:
        # Извлекаем текст
        if isinstance(item, dict):
            text = item.get('text', '')
            start = int(item.get('start', 0))
        else:
            text = getattr(item, 'text', str(item))
            start = int(getattr(item, 'start', 0))

        line = f"[{start}s] {text}"
        if total_length + len(line) > max_chars:
            break
        formatted_lines.append(line)
        total_length += len(line)

    return "\n".join(formatted_lines)


@app.post("/api/analyze")
async def analyze_video(request: AnalyzeRequest):
    video_id = request.video_id
    
    # 1. Получаем raw-субтитры с временными метками
    transcript_data = get_youtube_transcript_raw(video_id)
    formatted_transcript = format_transcript_with_timestamps(transcript_data)

    # 2. Промпт для ИИ с расширенными полями
    system_prompt = """
    Ты — эксперт по когнитивной психологии, критическому мышлению и логике.
    Твоя задача — проанализировать предоставленный транскрипт видео с таймкодами вида [Xs] и найти в нём когнитивные ошибки (искажения), манипуляции или логические неувязки.

    Отвечай строго в формате JSON следующей структуры:
    {
      "summary": "Краткое резюме об общем качестве аргументации в видео (1-2 предложения).",
      "credibility_score": 75, // Число от 0 до 100 (индекс аргументированности, где 100 - идеальная аргументация без ошибок)
      "biases": [
        {
          "name": "Название когнитивной ошибки/манипуляции (на русском)",
          "category": "Логическая ошибка" | "Эмоциональная манипуляция" | "Фальшивые факты" | "Подмена понятий",
          "quote": "Цитата из видео",
          "timestamp": 125, // Секунда, на которой это сказано (число из скобок [Xs])
          "explanation": "Подробное объяснение, почему это является ошибкой мышления в данном контексте"
        }
      ]
    }

    Категории ("category") выбирай строго из следующих четырех: "Логическая ошибка", "Эмоциональная манипуляция", "Фальшивые факты", "Подмена понятий".
    Если когнитивных ошибок не найдено, верни credibility_score: 100 и пустой массив "biases".
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Транскрипт видео с таймкодами:\n{formatted_transcript}"}
            ],
            temperature=0.2
        )
        
        # Парсим строка в JSON для уверенности в структуре ответа
        res_content = response.choices[0].message.content
        return json.loads(res_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка взаимодействия с Groq AI: {str(e)}")


@app.get("/")
def read_root():
    return {"status": "Server is running (Groq Backend v2)"}