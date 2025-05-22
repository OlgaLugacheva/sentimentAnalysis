# src/train_model_bert.py

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

def train_and_save_bert_model():
    # Загружаем и обрабатываем данные
    df = pd.read_csv("../data/sentimentdataset_2.csv")

    # Применение маппинга к меткам
    df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.capitalize()
    print("Уникальные значения после нормализации:", df['Sentiment'].unique())
    df['Sentiment'] = df['Sentiment'].map(emotion_to_sentiment)

    df = df.dropna(subset=["Text", "Sentiment"])  # Удаляем пропуски

    # Фильтрация редких классов
    value_counts = df["Sentiment"].value_counts()
    valid_labels = value_counts[value_counts >= 2].index
    df = df[df["Sentiment"].isin(valid_labels)].reset_index(drop=True)

    # Повторное кодирование меток
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["Sentiment"])

    # Разделение на обучающую и тестовую выборки
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label_encoded"],
        random_state=42,
    )

    # Токенизация
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(batch):
        tokens = tokenizer(batch["Text"], truncation=True, padding="max_length")
        tokens["labels"] = batch["label_encoded"]
        return tokens

    # Создание датасетов HuggingFace
    train_dataset = Dataset.from_pandas(train_df[["Text", "label_encoded"]])
    test_dataset = Dataset.from_pandas(test_df[["Text", "label_encoded"]])

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Загрузка модели
    num_labels = len(np.unique(df["label_encoded"]))
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    # Параметры обучения
    training_args = TrainingArguments(
        output_dir="../models/bert_model",
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

    # Обучение
    trainer.train()

    # Оценка
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    # Показываем только реально встречающиеся метки
    used_labels = np.unique(np.concatenate((y_true, y_pred)))
    target_names = label_encoder.inverse_transform(used_labels)

    print(classification_report(y_true, y_pred, labels=used_labels, target_names=target_names))

    # Сохранение модели и энкодера
    model.save_pretrained("../models/bert_model")
    tokenizer.save_pretrained("../models/bert_model")
    joblib.dump(label_encoder, "../models/label_encoder.pkl")

if __name__ == "__main__":
    train_and_save_bert_model()
