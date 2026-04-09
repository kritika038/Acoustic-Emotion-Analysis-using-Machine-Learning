import joblib
import numpy as np

from src.feature_extraction import extract_features


model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_map = joblib.load("models/labels.pkl")

inv_label_map = {v: k for k, v in label_map.items()}


def predict_emotion(file_path):
    features = extract_features(file_path)

    features = scaler.transform([features])

    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    emotion = inv_label_map[pred]

    return emotion, probs, list(inv_label_map.values())