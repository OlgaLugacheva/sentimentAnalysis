import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from pandas import DataFrame

# Маппинг эмоций в три класса
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

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Удаляем URL
    text = re.sub(r"http\S+", "", text)
    # Удаляем упоминания
    text = re.sub(r"@\w+", "", text)
    # Удаляем хэштеги
    text = re.sub(r"#\w+", "", text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def map_emotions(df):
    df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.lower()
    print("Уникальные значения после нормализации:", df['Sentiment'].unique())
    emotion_to_sentiment_clean = {
        key.strip().lower(): value for key, value in emotion_to_sentiment.items()
    }
    df['Sentiment'] = df['Sentiment'].map(emotion_to_sentiment_clean)
    return df

def load_and_preprocess_data(df : DataFrame):

    emotion_to_sentiment_clean = {
        key.strip().lower(): value for key, value in emotion_to_sentiment.items()
    }

    # Если есть колонка Sentiment в файле
    if "Sentiment" in df.columns:
        # Приводим к единому формату
        df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.lower()

        print("Уникальные значения после нормализации:", df['Sentiment'].unique())
        df['Sentiment'] = df['Sentiment'].map(emotion_to_sentiment_clean)

    df = df.drop_duplicates()

    # Удалим строки, где Text отсутствует
    df = df[df['Text'].notnull()]

    # Приведем всё к строкам
    df['Text'] = df['Text'].astype(str)
    df['Text_clean'] = df['Text'].apply(lambda x: clean_text(x))
    return df
