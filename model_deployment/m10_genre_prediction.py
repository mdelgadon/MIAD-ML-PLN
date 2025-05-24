
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
clf = joblib.load(os.path.join(BASE_DIR, 'genre_clf.pkl'))
vect = joblib.load(os.path.join(BASE_DIR, 'genre_vectorizer.pkl'))
le = joblib.load(os.path.join(BASE_DIR, 'genre_binarizer.pkl'))

def predict_genres(plot: str, threshold: float = 0.3) -> list:
    X_new = vect.transform([plot])
    proba = clf.predict_proba(X_new)[0]
    return [genre for genre, p in zip(le.classes_, proba) if p >= threshold]

def predict_genres_proba(plot: str) -> dict:
    X_new = vect.transform([plot])
    proba = clf.predict_proba(X_new)[0]
    
    # Conversión segura a tipos nativos de Python
    return {
        'p_' + genre: float(p) 
        for genre, p in zip(le.classes_, proba)
    }
