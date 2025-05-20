# Импорт необходимых библиотек:
# - pandas и numpy для работы с данными
# - matplotlib для визуализации
# - nltk для обработки текста (стоп-слова, стемминг, токенизация)
# - wordcloud для визуализации в виде облака слов
# - collections для подсчёта частоты элементов
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from collections import Counter

# Загрузка данных
df = pd.read_csv("C:/Users/anya8/Downloads/sentimentdataset.csv")

# Первичный осмотр данных
print(f"Размер датасета: {df.shape}\n")
print(f"Информация о датасете:")
df.info()
print(f"\nКоличество пропусков: {df.isnull().sum().sum()}\n")
print(f"Количество дубликатов: {df.duplicated().sum().sum()}")

# Словарь маппинга
emotion_to_sentiment = {
    # POSITIVE EMOTIONS
    "Admiration": "positive",
    "Approval": "positive",
    "Gratitude": "positive",
    "Optimism": "positive",
    "Love": "positive",
    "Excitement": "positive",
    "Joy": "positive",
    "Caring": "positive",
    "Happiness": "positive",
    "Enjoyment": "positive",
    "Affection": "positive",
    "Awe": "positive",
    "Adoration": "positive",
    "Pride": "positive",
    "Elation": "positive",
    "Euphoria": "positive",
    "Contentment": "positive",
    "Serenity": "positive",
    "Hope": "positive",
    "Empowerment": "positive",
    "Compassion": "positive",
    "Tenderness": "positive",
    "Relief": "positive",
    "Grateful": "positive",
    "Playful": "positive",
    "Inspired": "positive",
    "Confidence": "positive",
    "Accomplishment": "positive",
    "Wonderment": "positive",
    "Positivity": "positive",
    "Success": "positive",
    "Heartwarming": "positive",
    "Celebration": "positive",
    "Ecstasy": "positive",
    "Kindness": "positive",
    "Joyfulreunion": "positive",
    "ocean's freedom": "positive",
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

    # NEGATIVE EMOTIONS
    "Anger": "negative",
    "Disappointment": "negative",
    "Disapproval": "negative",
    "Disgust": "negative",
    "Fear": "negative",
    "Grief": "negative",
    "Annoyance": "negative",
    "Embarrassment": "negative",
    "Remorse": "negative",
    "Frustration": "negative",
    "Sadness": "negative",
    "Hate": "negative",
    "Despair": "negative",
    "Loss": "negative",
    "Jealousy": "negative",
    "Regret": "negative",
    "Betrayal": "negative",
    "Suffering": "negative",
    "Heartbreak": "negative",
    "Desperation": "negative",
    "Helplessness": "negative",
    "Angry": "negative",
    "Overwhelmed": "negative",
    "Embarrassed": "negative",
    "Envious": "negative",
    "Darkness": "negative",
    "Devastated": "negative",
    "Hurt": "negative",
    "Bad": "negative",
    "Shame": "negative",
    "Jealous": "negative",
    "Guilt": "negative",
    "Loneliness": "negative",
    "Resentment": "negative",
    "Envy": "negative",
    "Agony": "negative",
    "Worry": "negative",
    "Fearfulness": "negative",
    "Anxiety": "negative",
    "Grudge": "negative",
    "Pain": "negative",
    "Rage": "negative",
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

    # NEUTRAL EMOTIONS
    "Neutral": "neutral",
    "Confusion": "neutral",
    "Curiosity": "neutral",
    "Realization": "neutral",
    "Surprise": "neutral",
    "Nervousness": "neutral",
    "Amusement": "neutral",
    "Indifference": "neutral",
    "Pensive": "neutral",
    "Contemplation": "neutral",
    "Reflection": "neutral",
    "Melancholy": "neutral",
    "Ambivalence": "neutral",
    "Boredom": "neutral",
    "Calmness": "neutral",
    "Acceptance": "neutral",
    "Intrigue": "neutral",
    "Harmony": "neutral",
    "Tranquility": "neutral",
    "Observation": "neutral",
    "Energy": "neutral",
    "Interest": "neutral",
    "Awareness": "neutral",
    "Skepticism": "neutral",
    "Uncertainty": "neutral",
    "emotion": "neutral",
    "emotionalstorm": "neutral",
    "nostalgia": "neutral",
    "thrill": "neutral",
    "thrilling journey": "neutral",
    "whispers of the past": "neutral",
    "solitude": "neutral"
}

# Убираем лишние пробелы и изменяем регистр для ключей в словаре для маппинга и для значений в самом датафрейме
emotion_to_sentiment = {key.strip().lower(): value for key, value in emotion_to_sentiment.items()}
df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.lower()

# Применяем маппинг
df['Sentiment_mapping'] = df['Sentiment'].map(emotion_to_sentiment)
print("Уникальные значения после нормализации:", df['Sentiment_mapping'].unique())

# Проверим, не появились ли у нас пропуски после маппинга
print(f"\nКоличество пропусков: {df.isnull().sum().sum()}")

# Анализ распределения классов
sentiment_counts = df['Sentiment_mapping'].value_counts()
print(f"Распределение классов: {sentiment_counts}")

# Визуализируем распределение
sentiment_counts.plot(kind='bar', color=['green', 'red', 'yellow'])
plt.title('Распределение классов Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Количество')
plt.xticks(rotation=0)
plt.show()

# Загружаем стоп-слова и токенизатор
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    # Токенизация
    tokens = word_tokenize(text.lower())
    # Удаление стоп-слов и пунктуации
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Стемминг
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Применение очистки к тексту
df['Text_clean'] = df['Text'].apply(clean_text)

# Пример до и после очистки
print("\nПример до и после очистки")
print("ДО:", df['Text'].iloc[0])
print("ПОСЛЕ:", df['Text_clean'].iloc[0])

# Создадим WordCloud для каждого класса
def wordcloud_plot_class (df, Sentiment_class):
    Text_cloud = ' '.join(df[df['Sentiment_mapping'] == Sentiment_class]['Text_clean'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(Text_cloud)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Облако слов для класса {Sentiment_class}')
    plt.axis('off')
    plt.show()

# Визуализируем для каждого класса
wordcloud_plot_class(df, 'positive')
wordcloud_plot_class(df, 'negative')
wordcloud_plot_class(df, 'neutral')

# Анализ длины текстов
df['Text_length'] = df['Text_clean'].apply(len)

# Визуализация распределения длины текстов по классам
plt.figure(figsize=(10, 5))
for sentiment in df['Sentiment_mapping'].unique():
    subset = df[df['Sentiment_mapping'] == sentiment]
    plt.hist(subset['Text_length'], alpha=0.5, label=sentiment)

plt.title('Распределение длины текстов по классам')
plt.xlabel('Длина текста, симв.')
plt.ylabel('Количество')
plt.legend()
plt.show()

# Подсчет кол-ва слов и выведение топ-20
def get_top_words(df, sentiment, n=20):
    all_words = ' '.join(df[df['Sentiment_mapping'] == sentiment]['Text_clean']).split()
    return Counter(all_words).most_common(n)

# Визуализация топ-слов
for sentiment in df['Sentiment_mapping'].unique():
    top_words = get_top_words(df, sentiment)
    words, counts = zip(*top_words)

    plt.figure(figsize=(10, 5))
    plt.barh(words, counts)
    plt.title(f'Топ-20 слов для класса {sentiment}')
    plt.xlabel('Частота')
    plt.gca().invert_yaxis()
    plt.show()

# НАДО ВЫБРАТЬ, КАК ЛУЧШЕ - ПЕРВЫЙ ВАРИАНТ ВИЗУАЛИЗАЦИИ ВЫШЕ, ВТОРОЙ - НИЖЕ
def plot_sentiment_analysis(df, sentiment_class):
    # Создаем фигуру, которая будет состоять из облака слов и рядом график топ-20
    plt.figure(figsize=(20, 8))

    # Облако слов
    plt.subplot(1, 2, 1)
    text_cloud = ' '.join(df[df['Sentiment_mapping'] == sentiment_class]['Text_clean'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_cloud)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Облако слов для класса {sentiment_class}', fontsize=14)
    plt.axis('off')

    # Топ-20 слов
    plt.subplot(1, 2, 2)
    all_words = ' '.join(df[df['Sentiment_mapping'] == sentiment_class]['Text_clean']).split()
    top_words = Counter(all_words).most_common(20)
    words, counts = zip(*top_words)

    plt.barh(words, counts, color='skyblue')
    plt.title(f'Топ-20 слов для класса {sentiment_class}', fontsize=14)
    plt.xlabel('Частота', fontsize=12)
    plt.ylabel('Слова', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.show()

# Визуализируем для всех классов сразу
for sentiment in df['Sentiment_mapping'].unique():
    plot_sentiment_analysis(df, sentiment)

# Визуализируем по отдельности для каждого класса
plot_sentiment_analysis(df, 'positive')
plot_sentiment_analysis(df, 'negative')
plot_sentiment_analysis(df, 'neutral')