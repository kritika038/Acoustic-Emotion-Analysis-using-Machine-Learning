import librosa
import numpy as np

def analyze_audio(file_path):
    y, sr = librosa.load(file_path)

    # Normalize
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    duration = librosa.get_duration(y=y, sr=sr)

    energy = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    pitch = float(np.mean(librosa.yin(y, fmin=50, fmax=300)))

    silence_ratio = float(np.mean(np.abs(y) < 0.001))

    return {
        "duration": duration,
        "energy": energy,
        "pitch": pitch,
        "silence": silence_ratio
    }