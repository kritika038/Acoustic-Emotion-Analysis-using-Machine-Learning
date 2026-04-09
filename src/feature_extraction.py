import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)

        # Skip very short audio
        if len(audio) < sr:
            return None

        # Normalize
        audio = librosa.util.normalize(audio)

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        # Delta
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        combined = np.vstack((mfcc, delta, delta2))

        features = np.mean(combined.T, axis=0)

        return features

    except Exception as e:
        print(f"Error: {e}")
        return None