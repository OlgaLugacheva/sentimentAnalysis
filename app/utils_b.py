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

from src.data_pre_processing import load_and_preprocess_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, "../models/vectorizer_b.pkl")

model = joblib.load("models/best_model_b.pkl")
label_encoder = joblib.load("models/label_encoder_b.pkl")
vectorizer = joblib.load(VECTORIZER_PATH)


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

#todo; delete
    # emotion_to_sentiment_clean = {
    #     key.strip().lower(): value for key, value in emotion_to_sentiment.items()
    # }
    df = load_and_preprocess_data(df)

    # Если есть колонка Sentiment в файле
    if "Sentiment" in df.columns:
        # Приводим к единому формату
        df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.lower()

        # print("Уникальные значения после нормализации:", df['Sentiment'].unique())
        # df['Sentiment'] = df['Sentiment'].map(emotion_to_sentiment_clean)

    # Прогноз для каждой строки
    df["Predict_sentiment"] = df["Text_clean"].astype(str).apply(lambda text: predict_sentiment(TextInput(text=text)))

    # Подсчет точности, если есть колонка 'Sentiment'
    # todo: сделать аналогично в другом или вынести логику
    accuracy = None
    if "Sentiment" in df.columns:
        # Оставляем только строки, где 'Sentiment' не пустое
        mask = df["Sentiment"].notna()
        true_labels = df.loc[mask, "Sentiment"].astype(str)
        predicted_labels = df.loc[mask, "Predict_sentiment"].astype(str)

        # Вычисляем долю совпадений
        if not true_labels.empty:
            accuracy = (true_labels == predicted_labels).mean()

    # Преобразование обратно в CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return {
        "file": output.getvalue(),
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
    }