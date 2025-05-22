# app/utils_bert.py
import io

import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import os


# Пути к модели и энкодеру
MODEL_PATH = "models/bert_model"
ENCODER_PATH = "models/label_encoder.pkl"

# Загружаем модель, токенизатор и LabelEncoder один раз
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # переводим в режим предсказания

label_encoder = joblib.load(ENCODER_PATH)


emotion_to_sentiment = {
    # POSITIVE
    "Admiration": "positive", "Approval": "positive", "Gratitude": "positive",
    "Optimism": "positive", "Love": "positive", "Excitement": "positive",
    "Joy": "positive", "Caring": "positive", "Happiness": "positive",
    "Enjoyment": "positive", "Affection": "positive", "Awe": "positive",
    "Adoration": "positive", "Pride": "positive", "Elation": "positive",
    "Euphoria": "positive", "Contentment": "positive", "Serenity": "positive",
    "Hope": "positive", "Empowerment": "positive", "Compassion": "positive",
    "Tenderness": "positive", "Relief": "positive", "Grateful": "positive",
    "Playful": "positive", "Inspired": "positive", "Confidence": "positive",
    "Accomplishment": "positive", "Wonderment": "positive", "Positivity": "positive",
    "Success": "positive", "Heartwarming": "positive", "Celebration": "positive",
    "Ecstasy": "positive", "Kindness": "positive", "Joyfulreunion": "positive", "ocean's freedom": "positive",

    # NEGATIVE
    "Anger": "negative", "Disappointment": "negative", "Disapproval": "negative",
    "Disgust": "negative", "Fear": "negative", "Grief": "negative",
    "Annoyance": "negative", "Embarrassment": "negative", "Remorse": "negative",
    "Frustration": "negative", "Sadness": "negative", "Hate": "negative",
    "Despair": "negative", "Loss": "negative", "Jealousy": "negative",
    "Regret": "negative", "Betrayal": "negative", "Suffering": "negative",
    "Heartbreak": "negative", "Desperation": "negative", "Helplessness": "negative",
    "Angry": "negative", "Overwhelmed": "negative",
    "Embarrassed": "negative", "Envious": "negative", "Darkness": "negative",
    "Devastated": "negative", "Hurt": "negative",
    "Bad": "negative", "Shame": "negative", "Jealous": "negative", "Guilt": "negative",
    "Loneliness": "negative", "Resentment": "negative", "Envy": "negative",
    "Agony": "negative", "Worry": "negative", "Fearfulness": "negative",
    "Anxiety": "negative", "Grudge": "negative", "Pain": "negative", "Rage": "negative",
        "apprehensive": "negative",
    "bitter": "negative",
    "bitterness": "negative",
    "bittersweet": "negative",
    "desolation": "negative",
    "disappointed": "negative",
    "dismissive": "negative",
    "exhaustion": "negative",
    "fearful": "negative",
    "frustrated": "negative",
    "heartache": "negative",
    "intimidation": "negative",
    "isolation": "negative",
    "lostlove": "negative",
    "miscalculation": "negative",
    "mischievous": "negative",
    "negative": "negative",
    "numbness": "negative",
    "obstacle": "negative",
    "pressure": "negative",
    "ruins": "negative",
    "sad": "negative",
    "sorrow": "negative",
    "suspense": "negative",
    "yearning": "negative",
        "adrenaline": "positive",
    "adventure": "positive",
    "amazement": "positive",
    "anticipation": "positive",
    "appreciation": "positive",
    "arousal": "positive",
    "artisticburst": "positive",
    "blessed": "positive",
    "breakthrough": "positive",
    "captivation": "positive",
    "celestial wonder": "positive",
    "challenge": "positive",
    "charm": "positive",
    "colorful": "positive",
    "compassionate": "positive",
    "confident": "positive",
    "connection": "positive",
    "coziness": "positive",
    "creative inspiration": "positive",
    "creativity": "positive",
    "culinary adventure": "positive",
    "culinaryodyssey": "positive",
    "dazzle": "positive",
    "determination": "positive",
    "dreamchaser": "positive",
    "elegance": "positive",
    "empathetic": "positive",
    "enchantment": "positive",
    "engagement": "positive",
    "enthusiasm": "positive",
    "envisioning history": "positive",
    "exploration": "positive",
    "festivejoy": "positive",
    "free-spirited": "positive",
    "freedom": "positive",
    "friendship": "positive",
    "fulfillment": "positive",
    "grandeur": "positive",
    "happy": "positive",
    "hopeful": "positive",
    "hypnotic": "positive",
    "iconic": "positive",
    "imagination": "positive",
"immersion": "positive",
    "innerjourney": "positive",
    "inspiration": "positive",
    "journey": "positive",
    "joy in baking": "positive",
    "kind": "positive",
    "marvel": "positive",
    "melodic": "positive",
    "mesmerizing": "positive",
    "mindfulness": "positive",
    "motivation": "positive",
    "nature's beauty": "positive",
    "overjoyed": "positive",
    "playfuljoy": "positive",
    "positive": "positive",
    "proud": "positive",
    "radiance": "positive",
    "rejuvenation": "positive",
    "renewed effort": "positive",
    "resilience": "positive",
    "reverence": "positive",
    "romance": "positive",
    "runway creativity": "positive",
    "satisfaction": "positive",
    "solace": "positive",
    "spark": "positive",
    "sympathy": "positive",
    "touched": "positive",
    "triumph": "positive",
    "vibrancy": "positive",
    "whimsy": "positive",
    "winter magic": "positive",
    "wonder": "positive",
    "zest": "positive",

    # NEUTRAL
    "Neutral": "neutral", "Confusion": "neutral", "Curiosity": "neutral",
    "Realization": "neutral", "Surprise": "neutral", "Nervousness": "neutral",
    "Amusement": "neutral", "Indifference": "neutral", "Pensive": "neutral",
    "Contemplation": "neutral", "Reflection": "neutral", "Melancholy": "neutral",
    "Ambivalence": "neutral", "Boredom": "neutral", "Calmness": "neutral",
    "Acceptance": "neutral", "Intrigue": "neutral", "Harmony": "neutral",
    "Tranquility": "neutral", "Observation": "neutral", "Energy": "neutral",
    "Interest": "neutral", "Awareness": "neutral", "Skepticism": "neutral",
    "Uncertainty": "neutral",
    "emotion": "neutral",
    "emotionalstorm": "neutral",
    "nostalgia": "neutral",
    "thrill": "neutral",
    "thrilling journey": "neutral",
    "whispers of the past": "neutral",
    "solitude": "neutral"
}

class TextInput(BaseModel):
    text: str

def predict_sentiment(input_data):
    text = input_data.text

    # Токенизация
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Предсказание
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    # Преобразуем индекс обратно в метку
    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
    return predicted_label
def predict_sentiment_batch(filepath_or_buffer: bytes):
    # Чтение CSV в DataFrame
    df = pd.read_csv(io.BytesIO(filepath_or_buffer))

    # Проверка наличия нужной колонки
    if "Text" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'Text' column")
    # sentiment check

    emotion_to_sentiment_clean = {
        key.strip().lower(): value for key, value in emotion_to_sentiment.items()
    }

    # Если есть колонка Sentiment в файле
    if "Sentiment" in df.columns:
        # Приводим к единому формату
        df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.lower()

        print("Уникальные значения после нормализации:", df['Sentiment'].unique())
        df['Sentiment'] = df['Sentiment'].map(emotion_to_sentiment_clean)

    # Прогноз для каждой строки
    df["Predict_sentiment"] = df["Text"].astype(str).apply(lambda text: predict_sentiment(TextInput(text=text)))

    # Подсчет точности, если есть колонка 'Sentiment'
    accuracy = None
    if "Sentiment" in df.columns:
        # Убедимся, что обе колонки имеют одинаковый тип (str)
        true_labels = df["Sentiment"].astype(str)
        predicted_labels = df["Predict_sentiment"].astype(str)

        # Вычисляем долю совпадений
        accuracy = (true_labels == predicted_labels).mean()

    # Преобразование обратно в CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return {
        "file": output.getvalue(),
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
    }