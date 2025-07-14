#!/usr/bin/env python3
"""
Enhanced Machine Learning Service for Spam Detection
Features: Improved feature extraction, better preprocessing, enhanced model
"""

import sys
import json
import joblib
import numpy as np
import os
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
import nltk
from scipy.sparse import hstack

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class SpamDetector:
    def __init__(self, spam_keywords=None, promo_keywords=None, urgency_keywords=None):
        self.model = None
        self.vectorizer = None
        from feature_utils import preprocess_text, extract_features
        self.preprocess_text = preprocess_text
        self.extract_additional_features = lambda text: extract_features(
            text,
            spam_keywords=spam_keywords,
            promo_keywords=promo_keywords,
            urgency_keywords=urgency_keywords
        )
        self.spam_keywords = spam_keywords
        self.promo_keywords = promo_keywords
        self.urgency_keywords = urgency_keywords
        self.load_model()

    def load_model(self):
        """Load trained model and vectorizer"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl')
            vectorizer_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')

            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                print("✅ Model and vectorizer loaded successfully")
            else:
                print("⚠  Model files not found, creating default model...")
                self.create_default_model()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.create_default_model()

    def load_default_model(self):
        """Manually reset to default model"""
        self.create_default_model()

    def create_default_model(self):
        """Fallback default model: expects external training data to be provided via CSV."""
        print("⚠ No model found and no CSV provided. Please upload a CSV to train the model.")
        self.model = None
        self.vectorizer = None

    # Shared feature functions are now set in __init__

    def predict(self, message):
        """Predict message type (spam or ham)"""
        try:
            if not self.model or not self.vectorizer:
                return {'prediction': 'error', 'confidence': 0.0, 'error': 'Model not loaded'}

            cleaned = self.preprocess_text(message)
            tfidf = self.vectorizer.transform([cleaned])
            extra = self.extract_additional_features(message).reshape(1, -1)
            X = hstack([tfidf, extra])

            pred = self.model.predict(X)[0]
            prob = self.model.predict_proba(X)[0]

            try:
                blob = TextBlob(message)
                sentiment = blob.sentiment.polarity
            except:
                sentiment = 0.0

            # Use the actual keyword lists in use (from self)
            features = {
                'length': len(message),
                'word_count': len(message.split()),
                'uppercase_count': sum(1 for c in message if c.isupper()),
                'exclamation_count': message.count('!'),
                'question_count': message.count('?'),
                'sentiment': sentiment,
                'spam_keywords': sum(1 for word in (self.spam_keywords or []) if word in message.lower()),
                'promo_keywords': sum(1 for word in (self.promo_keywords or []) if word in message.lower()),
                'urgency_keywords': sum(1 for word in (self.urgency_keywords or []) if word in message.lower())
            }

            return {
                'prediction': 'spam' if pred == 1 else 'ham',
                'confidence': float(prob[pred]),
                'features': features,
                'probabilities': {'spam': float(prob[1]), 'ham': float(prob[0])}
            }

        except Exception as e:
            return {'prediction': 'error', 'confidence': 0.0, 'error': str(e)}

    def get_model_info(self):
        """Return model details"""
        if not self.model or not self.vectorizer:
            return {'loaded': False}

        try:
            return {
                'loaded': True,
                'model_type': str(type(self.model)._name_),
                'features': self.vectorizer.get_feature_names_out().tolist()[:10],
                'feature_count': len(self.vectorizer.get_feature_names_out())
            }
        except:
            return {'loaded': False}

def main():
    """Command line prediction"""
    if len(sys.argv) != 2:
        print("Usage: python ml_service.py '<message>'")
        sys.exit(1)

    message = sys.argv[1]
    detector = SpamDetector()
    result = detector.predict(message)
    print(json.dumps(result, indent=2))

if __name__ == '_main_':
    main()