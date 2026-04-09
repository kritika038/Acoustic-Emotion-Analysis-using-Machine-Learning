import librosa
import soundfile as sf
import tempfile

from src.feature_extraction import extract_features


def predict_timeline(file_path, model, scaler, inv_label_map):
    audio, sr = librosa.load(file_path)

    chunk_size = sr  # 1 second
    timeline = []

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]

        if len(chunk) < chunk_size:
            continue

        # Save chunk safely
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, chunk, sr)

            features = extract_features(tmp.name)

            if features is None:
                continue

            features = scaler.transform([features])

            pred = model.predict(features)[0]
            emotion = inv_label_map[pred]

            timeline.append(emotion)

    return timeline