import re
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Ensure nltk dependencies are downloaded
nltk.download('punkt', quiet=True)

def preprocess_text(text: str) -> str:
    """
    Cleans and preprocesses text: lowercasing, removing special chars, stopwords.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)

def extract_features(text: str) -> dict:
    """
    Extracts custom features from text: length, sentiment, spam keywords.
    """
    clean_text = preprocess_text(text)

    # Sentiment using TextBlob
    sentiment = TextBlob(clean_text).sentiment.polarity

    # Custom keyword checks
    spam_keywords = ["free", "win", "offer", "buy", "urgent", "money", "click"]
    keyword_count = sum(word in clean_text for word in spam_keywords)

    return {
        "clean_text": clean_text,
        "length": len(clean_text.split()),
        "sentiment": sentiment,
        "spam_keyword_count": keyword_count
    }