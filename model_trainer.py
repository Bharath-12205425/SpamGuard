import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from scipy.sparse import hstack
import time
import re

# Download required NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class ModelTrainer:
    def __init__(self, spam_keywords=None, promo_keywords=None, urgency_keywords=None):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
        )
        self.spam_keywords = spam_keywords
        self.promo_keywords = promo_keywords
        self.urgency_keywords = urgency_keywords

    def preprocess(self, text):
        from feature_utils import preprocess_text
        return preprocess_text(text)

    def extract_features(self, text):
        from feature_utils import extract_features
        return extract_features(
            text,
            spam_keywords=self.spam_keywords,
            promo_keywords=self.promo_keywords,
            urgency_keywords=self.urgency_keywords
        )

    def train_model(self, data):
        try:
            start = time.time()
            df = pd.DataFrame(data)
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            df['cleaned'] = df['message'].apply(self.preprocess)

            # Fit the vectorizer and transform the cleaned text
            X_tfidf = self.vectorizer.fit_transform(df['cleaned'])

            # Extract custom features
            extra_features = np.array([self.extract_features(msg) for msg in df['message']])

            # Combine TF-IDF and additional features
            X_combined = hstack([X_tfidf, extra_features])
            y = df['label']

            # Split dataset
            if len(df) > 4:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_combined, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = X_combined, X_combined, y, y

            # Train model
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            metrics = {
                'accuracy': round(accuracy_score(y_test, y_pred), 4),
                'precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
                'recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
                'f1_score': round(f1_score(y_test, y_pred, average='weighted'), 4)
            }

            # Save model and vectorizer
            joblib.dump(model, 'trained_model.pkl')
            joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')

            return {
                'success': True,
                'metrics': metrics,
                'training_time': round(time.time() - start, 2)
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}