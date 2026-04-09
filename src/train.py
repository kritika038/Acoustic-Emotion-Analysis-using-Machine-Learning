import os
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from collections import Counter

from xgboost import XGBClassifier

from src.feature_extraction import extract_features


DATA_PATH = "data"


def load_data():
    X, y = [], []

    print("📂 Loading dataset...")

    for label in os.listdir(DATA_PATH):
        folder = os.path.join(DATA_PATH, label)

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            if file.endswith(".wav"):
                path = os.path.join(folder, file)

                features = extract_features(path)

                if features is not None:
                    X.append(features)
                    y.append(label)

    return np.array(X), np.array(y)


def train():
    X, y = load_data()

    print(f"✅ Loaded {len(X)} samples")
    print("📊 Class Distribution:", Counter(y))

    # Encode labels
    label_map = {label: idx for idx, label in enumerate(sorted(set(y)))}
    y_encoded = np.array([label_map[label] for label in y])

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Model
    model = XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=3,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_lambda=1,
    scale_pos_weight=1,
    objective="multi:softprob",
    eval_metric="mlogloss"
)

    # Cross Validation
    print("🔁 Running Cross Validation...")
    scores = cross_val_score(model, X, y_encoded, cv=5)
    print(f"📊 CV Accuracy: {scores.mean():.4f}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("🤖 Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"🎯 Test Accuracy: {acc:.4f}")

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_map, "models/labels.pkl")

    print("💾 Model saved")


if __name__ == "__main__":
    train()