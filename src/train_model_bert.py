
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch
from datasets import Dataset
import os

from src.data_pre_processing import clean_text
from src.data_pre_processing import map_emotions

MODELS_DIR = "models"
MODEL_DIR = "models/bert_model"
MODEL_DIR_TUNED = "models/bert_model_tuned"
ENCODER_PATH = "models/label_encoder.pkl"

"""
Загружает токенизатор и либо:
Загружает уже обученную модель из MODEL_DIR, если она существует.
Иначе — загружает базовую bert-base-uncased модель и задаёт количество меток (num_labels) для классификации.

"""
def load_tokenizer_and_model(num_labels=None):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if os.path.exists(MODEL_DIR):
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    else:
        assert num_labels is not None, "num_labels must be provided when training from scratch"
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    return tokenizer, model

"""Очищает текст, кодирует метки .

Токенизирует текст и добавляет метку (labels) для обучения."""

def tokenize_dataset(df, tokenizer, label_encoder):
    df["Text_clean"] = df["Text"].apply(lambda x: clean_text(x))
    df["label_encoded"] = label_encoder.transform(df["Sentiment"])
    dataset = Dataset.from_pandas(df[["Text_clean", "label_encoded"]])

    def tokenize_function(batch):
        tokens = tokenizer(batch["Text_clean"], truncation=True, padding="max_length")
        tokens["labels"] = batch["label_encoded"]
        return tokens

    dataset = dataset.map(tokenize_function, batched=True)
    return dataset

"""
Загружает CSV-файл с датасетом.
Преобразует эмоции в метки (map_emotions) и очищает текст (clean_text).
Удаляет слишком редкие метки (меньше 2 наблюдений).
Кодирует метки с помощью LabelEncoder.
Делит данные на обучающую и тестовую выборки.
Загружает/инициализирует модель и токенизатор.
Определяет параметры обучения
Обучает модель с помощью Trainer.
Сохраняет модель, токенизатор и энкодер меток

"""

def train_and_save_bert_model():
    df = pd.read_csv("../data/sentimentdataset_2.csv").drop_duplicates()
    df = map_emotions(df).dropna(subset=["Text", "Sentiment"])
    df['Text_clean'] = df['Text'].apply(clean_text)

    value_counts = df["Sentiment"].value_counts()
    valid_labels = value_counts[value_counts >= 2].index
    df = df[df["Sentiment"].isin(valid_labels)].reset_index(drop=True)

    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["Sentiment"])

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label_encoded"], random_state=42)

    tokenizer, model = load_tokenizer_and_model(num_labels=len(label_encoder.classes_))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = tokenize_dataset(train_df, tokenizer, label_encoder)
    test_dataset = tokenize_dataset(test_df, tokenizer, label_encoder)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="../logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    used_labels = np.unique(np.concatenate((y_true, y_pred)))
    print(classification_report(y_true, y_pred, labels=used_labels, target_names=label_encoder.inverse_transform(used_labels)))

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    joblib.dump(label_encoder, ENCODER_PATH)

"""
Дообучает уже натренированную модель на новых данных.
"""
def fine_tune_model_on_new_data(new_data: pd.DataFrame, epochs: int = 1):
    label_encoder: LabelEncoder = joblib.load(ENCODER_PATH)
    tokenizer, model = load_tokenizer_and_model()
    # Заморозка весов BERT
    for param in model.bert.parameters():
        param.requires_grad = False
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Загружаем оригинальный датасет
    original_data = pd.read_csv("data/sentimentdataset_2.csv").drop_duplicates()
    original_data = map_emotions(original_data).dropna(subset=["Text", "Sentiment"])

    # Объединяем с новым (ошибочным) датасетом
    combined_data = pd.concat([original_data, new_data], ignore_index=True)
    combined_data = combined_data.dropna(subset=["Text", "Sentiment"])
    combined_data = combined_data[combined_data["Sentiment"].isin(label_encoder.classes_)]

    if combined_data.empty:
        raise ValueError("Нет допустимых данных для дообучения")

    combined_data["Text_clean"] = combined_data["Text"].apply(clean_text)
    combined_data["label_encoded"] = label_encoder.transform(combined_data["Sentiment"])

    train_dataset = tokenize_dataset(combined_data, tokenizer, label_encoder)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=4,
        num_train_epochs=epochs,
        logging_dir="../logs",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(MODEL_DIR_TUNED)
    tokenizer.save_pretrained(MODEL_DIR_TUNED)
    print("Модель успешно дообучена и сохранена.")


if __name__ == "__main__":
    train_and_save_bert_model()
