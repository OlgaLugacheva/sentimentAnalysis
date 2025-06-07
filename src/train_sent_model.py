from collections import Counter

import joblib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud

from src.data_pre_processing import clean_text

MODEL_DIR_TUNED = "models/stack_model_tuned"

"""
Строит визуализацию для заданного класса тональности — облако слов и гистограмму 20 самых частотных слов.
"""
def plot_sentiment_analysis(df, sentiment_class):
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    text_cloud = ' '.join(df[df['Sentiment'] == sentiment_class]['Text_clean'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_cloud)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Облако слов для класса {sentiment_class}', fontsize=14)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    all_words = ' '.join(df[df['Sentiment'] == sentiment_class]['Text_clean']).split()
    top_words = Counter(all_words).most_common(20)
    words, counts = zip(*top_words)
    plt.barh(words, counts, color='skyblue')
    plt.title(f'Топ-20 слов для класса {sentiment_class}', fontsize=14)
    plt.xlabel('Частота', fontsize=12)
    plt.ylabel('Слова', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

"""
Очищает, кодирует и разбивает данные на обучающую и тестовую выборки. Применяет TF-IDF векторизацию.
"""
def tfidf_vectorizer_split(df, label_encoder, text_column='Text_clean', target_column='Sentiment', max_features=5000):
    X = df[text_column]
    y = df[target_column]

    y = label_encoder.fit_transform(df["Sentiment"])
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        if len(set(y_test)) < len(set(y)):
            print("Предупреждение: не все классы попали в тестовую выборку при стратификации.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=min(0.5, (len(y.unique()) * 2 + 1) / len(y)),
                random_state=42, stratify=y)
    except ValueError as e:
        print(f"Ошибка при train_test_split: {e}")
        print("Используем простой split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"\nРазмер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    print(f"Распределение классов в y_test: {pd.Series(y_test).value_counts().to_dict()}")
    print(f"Уникальные классы в y_test: {sorted(list(set(y_test)))}")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

"""
Назначение: перебирает комбинации из двух базовых моделей + одну финальную, обучает стекинг и выбирает лучший по accuracy.
Обучает базовые модели (логистическая регрессия, наивный Байес, случайный лес) и создает стекинг-модели с различными параметрами.

"""
def train_models(X_train_tfidf, X_test_tfidf, y_train, y_test):
    global_best_model = None
    global_best_acc = -1

    base_model_1 = ('lr', LogisticRegression(solver='liblinear', multi_class='ovr', C=1.0, random_state=42))
    base_model_2 = ('nb', MultinomialNB(alpha=0.1))
    base_model_3 = ('rf', RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1))

    models = [base_model_1, base_model_2, base_model_3]

    print("\n--- Индивидуальные предсказания базовых моделей ---")
    for name, model in models:
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        print(f"\nМодель: {name}")
        print(f"Точность: {accuracy_score(y_test, preds):.4f}")
        print(classification_report(y_test, preds, labels=model.classes_,
                                    target_names=[str(c) for c in model.classes_], zero_division=0))

    def run_stacking(name, base_models, final_model, passthrough):
        nonlocal global_best_model, global_best_acc
        print(f"\n--- Stacking: {name} ---")
        stacker = StackingClassifier(
            estimators=base_models,
            final_estimator=final_model,
            cv=5,
            n_jobs=-1,
            passthrough=passthrough
        )
        try:
            stacker.fit(X_train_tfidf, y_train)
            y_pred = stacker.predict(X_test_tfidf)
            acc = accuracy_score(y_test, y_pred)
            print(f"Точность: {acc:.4f}")
            if acc > global_best_acc:
                global_best_acc = acc
                global_best_model = stacker
            print(classification_report(
                y_test, y_pred,
                labels=stacker.classes_,
                target_names=[str(c) for c in stacker.classes_],
                zero_division=0
            ))
        except Exception as e:
            print(f"Ошибка Stacking {name}: {e}")

    meta = LogisticRegression(solver='liblinear', random_state=42)

    run_stacking("Set1_Passthrough_True", models, meta, True)
    run_stacking("Set1_Passthrough_False", models, meta, False)
    run_stacking("Set2_Passthrough_True", [base_model_1, base_model_3], meta, True)
    run_stacking("Set2_Passthrough_False", [base_model_1, base_model_3], meta, False)
    run_stacking("Set3_Passthrough_True", [base_model_1], meta, True)

    print("\n--- Лучшая модель ---")
    print(global_best_model)
    return global_best_model

"""
 Назначение: запускает весь пайплайн обучения: загрузка, очистка, визуализация, векторизация, обучение и сохранение лучшей модели
"""
def train_and_save_model():

    df = pd.read_csv("../data/Tweets.csv")
    df = df.drop_duplicates()
    sentiment_counts = df['Sentiment'].value_counts()
    print(f"Распределение классов: {sentiment_counts}")

    sentiment_counts.plot(kind='bar', color=['green', 'red', 'yellow'])
    plt.title('Распределение классов Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Количество')
    plt.xticks(rotation=0)
    plt.show()


    df['Text_clean'] = df['Text'].apply(lambda x: clean_text(x))

    print("\nПример до и после очистки")
    print("ДО:", df['Text'].iloc[0])
    print("ПОСЛЕ:", df['Text_clean'].iloc[0])

    print(df['Text_clean'].unique())
    print(f"\nКоличество пропусков: {df.isnull().sum()}\n")

    df['Text_length'] = df['Text_clean'].apply(len)

    plt.figure(figsize=(10, 5))
    for sentiment in df['Sentiment'].unique():
        subset = df[df['Sentiment'] == sentiment]
        plt.hist(subset['Text_length'], alpha=0.5, label=sentiment)

    plt.title('Распределение длины текстов по классам')
    plt.xlabel('Длина текста, симв.')
    plt.ylabel('Количество')
    plt.legend()
    plt.show()

    df = df[df['Text_length'] >= 10]

    plt.figure(figsize=(10, 5))
    for sentiment in df['Sentiment'].unique():
        subset = df[df['Sentiment'] == sentiment]
        plt.hist(subset['Text_length'], alpha=0.5, label=sentiment)

    plt.title('Распределение длины текстов по классам')
    plt.xlabel('Длина текста, симв.')
    plt.ylabel('Количество')
    plt.legend()
    plt.show()
    label_encoder = LabelEncoder()
    for sentiment in df['Sentiment'].unique():
        plot_sentiment_analysis(df, sentiment)

    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = tfidf_vectorizer_split(df, label_encoder)
    best_model = train_models(X_train_tfidf, X_test_tfidf, y_train, y_test)

    if best_model:
        filename_model = "../models/best_model_b.pkl"
        filename_vector = "../models/vectorizer_b.pkl"
        joblib.dump(best_model, filename_model)
        joblib.dump(vectorizer, filename_vector)
        joblib.dump(label_encoder, "../models/label_encoder_b.pkl")
        print(f"Модель сохранена в {filename_model}")
        print(f"Векторизатор сохранен в {filename_vector}")

"""
Назначение: переобучает стек-модель на новых данных, комбинируя с предыдущими. Это позволяет делать "дообучение", а не тренировать с нуля.
"""
def fine_tune_model_on_new_data(new_data: pd.DataFrame):
    # Загрузка энкодера и моделей
    label_encoder: LabelEncoder = joblib.load("models/label_encoder_b.pkl")
    vectorizer: TfidfVectorizer = joblib.load("models/vectorizer_b.pkl")
    old_model: StackingClassifier = joblib.load("models/best_model_b.pkl")

    # Загрузка оригинального датасета
    original_data = pd.read_csv("data/sentimentdataset_2.csv").drop_duplicates()
    original_data = original_data.dropna(subset=["Text", "Sentiment"])
    original_data["Text_clean"] = original_data["Text"].apply(clean_text)

    # Подготовка новых данных
    new_data = new_data.dropna(subset=["Text", "Sentiment"])
    new_data["Text_clean"] = new_data["Text"].apply(clean_text)

    # Объединение
    combined_data = pd.concat([
        original_data[['Text_clean', 'Sentiment']],
        new_data[['Text_clean', 'Sentiment']]
    ], ignore_index=True)
    combined_data = combined_data.dropna(subset=["Text_clean", "Sentiment"])

    combined_data = combined_data[combined_data["Sentiment"].isin(label_encoder.classes_)]
    if len(combined_data) < 50:
        raise ValueError(
            "Слишком мало данных после фильтрации по меткам. Возможно, в новых данных неизвестные классы.")

    # Повторная векторизация и энкодинг
    X = combined_data["Text_clean"]
    y = label_encoder.transform(combined_data["Sentiment"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Обучаем новый векторизатор
    new_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    X_train_tfidf = new_vectorizer.fit_transform(X_train)
    X_test_tfidf = new_vectorizer.transform(X_test)

    # Переобучаем стек-модель
    base_model_1 = ('lr', LogisticRegression(solver='liblinear', multi_class='ovr', C=1.0, random_state=42))
    base_model_2 = ('nb', MultinomialNB(alpha=0.1))
    base_model_3 = ('rf', RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1))

    meta_model = LogisticRegression(solver='liblinear', random_state=42)

    tuned_model = StackingClassifier(
        estimators=[base_model_1, base_model_2, base_model_3],
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1,
        passthrough=True
    )

    tuned_model.fit(X_train_tfidf, y_train)
    y_pred = tuned_model.predict(X_test_tfidf)
    print("Дообученная модель:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in label_encoder.classes_]))

    # Сохраняем обновлённые артефакты
    joblib.dump(tuned_model, "models/stack_model_tuned.pkl")
    joblib.dump(new_vectorizer, "models/vectorizer_tuned.pkl")
    joblib.dump(label_encoder, "models/label_encoder_tuned.pkl")

    print("Стек-модель успешно дообучена и сохранена.")


if __name__ == "__main__":
    train_and_save_model()
