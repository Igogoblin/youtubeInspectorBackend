import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI

app = FastAPI(title="YouTube Bias Detector API (Groq)")

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
    transcript_text: Optional[str] = None  # Принимаем текст субтитров напрямую от браузера


def get_youtube_transcript_fallback(video_id: str) -> str:
    """Резервное извлечение субтитров на сервере (если клиент не прислал текст)"""
    try:
        data = YouTubeTranscriptApi.get_transcript(video_id, languages=['ru', 'ru-RU', 'en', 'en-US'])
        formatted_lines = []
        for item in data:
            start = int(item.get('start', 0))
            text = item.get('text', '')
            formatted_lines.append(f"[{start}s] {text}")
        return "\n".join(formatted_lines)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось получить субтитры на сервере (IP заблокирован YouTube): {str(e)}"
        )


@app.post("/api/analyze")
async def analyze_video(request: AnalyzeRequest):
    video_id = request.video_id
    
    # 1. Используем текст от расширения или делаем fallback-запрос
    if request.transcript_text and request.transcript_text.strip():
        formatted_transcript = request.transcript_text
    else:
        formatted_transcript = get_youtube_transcript_fallback(video_id)

    # Ограничение по длине текста (~15 000 символов)
    max_chars = 15000
    if len(formatted_transcript) > max_chars:
        formatted_transcript = formatted_transcript[:max_chars] + "..."

    system_prompt = """
    Ты — эксперт по когнитивной психологии, критическому мышлению и логике.
    Твоя задача — проанализировать предоставленный транскрипт видео с таймкодами вида [Xs] и найти в нём когнитивные ошибки (искажения), манипуляции или логические неувязки.

    Отвечай строго в формате JSON следующей структуры:
    {
      "summary": "Краткое резюме об общем качестве аргументации в видео (1-2 предложения).",
      "credibility_score": 75,
      "biases": [
        {
          "name": "Название когнитивной ошибки/манипуляции (на русском)",
          "category": "Логическая ошибка" | "Эмоциональная манипуляция" | "Фальшивые факты" | "Подмена понятий",
          "quote": "Цитата из видео",
          "timestamp": 125,
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
        
        res_content = response.choices[0].message.content
        return json.loads(res_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка взаимодействия с Groq AI: {str(e)}")


@app.get("/")
def read_root():
    return {"status": "Server is running (Client-side transcript support)"}