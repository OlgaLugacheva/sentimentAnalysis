import io
import os
import pandas as pd
import torch
import joblib
from fastapi import HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data_pre_processing import load_and_preprocess_data
# Пути к модели и энкодеру
MODEL_PATH = "models/bert_model"
ENCODER_PATH = "models/label_encoder.pkl"

# Динамическая загрузка модели, токенизатора и энкодера
def load_model_components(model_version: str = "_origin"):

    model_version_d = "bert_model" if model_version == "_origin" else "bert_model_tuned"
    model_path = os.path.join("models", model_version_d)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель '{model_version}' не найдена по пути {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    label_encoder = joblib.load(ENCODER_PATH)

    return tokenizer, model, label_encoder


class TextInput(BaseModel):
    text: str
    model_version: str = "bert_model"  # По умолчанию используем старую модель


def predict_sentiment(input_data: TextInput, model_version: str = "bert_model"):
    tokenizer, model, label_encoder = load_model_components(model_version)

    inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
    return predicted_label


def predict_sentiment_batch(filepath_or_buffer: bytes, model_version: str = "bert_model"):
    tokenizer, model, label_encoder = load_model_components(model_version)

    df = pd.read_csv(io.BytesIO(filepath_or_buffer))

    if "Text" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'Text' column")

    df = load_and_preprocess_data(df)

    def predict_row(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
        return label_encoder.inverse_transform([predicted_class_id])[0]

    df["Predict_sentiment"] = df["Text_clean"].astype(str).apply(predict_row)

    accuracy = None
    if "Sentiment" in df.columns:
        true_labels = df["Sentiment"].astype(str)
        predicted_labels = df["Predict_sentiment"].astype(str)
        accuracy = (true_labels == predicted_labels).mean()

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return {
        "file": output.getvalue(),
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
    }
