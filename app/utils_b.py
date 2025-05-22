# utils
import os
import joblib
import pandas as pd
from fastapi import File, HTTPException
import io

from pandas._typing import ReadCsvBuffer
from tensorflow.keras.models import load_model
import numpy as np
from pydantic import BaseModel, FilePath

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, "../models/vectorizer_b.pkl")

model = joblib.load("models/best_model_b.pkl")
label_encoder = joblib.load("models/label_encoder_b.pkl")
vectorizer = joblib.load(VECTORIZER_PATH)

# Маппинг эмоций в три класса

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


def predict_sentiment(input_data: TextInput):
    processed_text = vectorizer.transform([input_data.text])

    # Получение вероятностей — опционально
    prediction_probs = model.predict_proba(processed_text.toarray())
    prediction_class = np.argmax(prediction_probs, axis=1)

    prediction_label = label_encoder.inverse_transform([prediction_class[0]])[0]
    return prediction_label

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