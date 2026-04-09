import streamlit as st
import numpy as np
import tempfile
import joblib
import json

from src.feature_extraction import extract_features
from src.audio_analysis import analyze_audio


# =========================
# LOAD MODEL
# =========================
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_map = joblib.load("models/labels.pkl")

inv_label_map = {v: k for k, v in label_map.items()}

emoji_map = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐"
}


# =========================
# PREDICT FUNCTION
# =========================
def predict(file_path):
    features = extract_features(file_path)
    features = scaler.transform([features])

    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    emotion = inv_label_map[pred]
    return emotion, probs


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Audio Behavior AI", page_icon="🎙️")

st.title("🎙️ Audio Behavior Analysis System")
st.markdown("### ML-based Emotion + Acoustic Intelligence")
st.markdown("---")


# =========================
# INPUT
# =========================
uploaded_file = st.file_uploader("📂 Upload WAV file", type=["wav"])
audio_input = st.audio_input("🎤 Record Voice")

audio_file = uploaded_file if uploaded_file else audio_input


# =========================
# PROCESS
# =========================
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        file_path = tmp.name

    st.audio(audio_file)

    with st.spinner("🔍 Analyzing audio..."):
        try:
            emotion, probs = predict(file_path)
            confidence = float(np.max(probs))

            # =========================
            # RESULT
            # =========================
            st.success(f"🎯 Emotion: {emotion} {emoji_map.get(emotion, '')}")

            st.progress(confidence)
            st.write(f"Confidence: {confidence:.2f}")

            if confidence > 0.75:
                st.success("High confidence prediction")
            elif confidence > 0.5:
                st.warning("Moderate confidence")
            else:
                st.error("Low confidence — prediction uncertain")

            # =========================
            # AUDIO ANALYSIS
            # =========================
            st.subheader("📊 Audio Insights")

            analysis = analyze_audio(file_path)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Duration", f"{analysis['duration']:.2f}s")
                st.metric("Energy", f"{analysis['energy']:.4f}")

            with col2:
                st.metric("Pitch", f"{analysis['pitch']:.2f} Hz")
                st.metric("Silence", f"{analysis['silence']:.2f}")

            # =========================
            # INTERPRETATION
            # =========================
            st.subheader("🧠 Interpretation")

            if emotion == "sad":
                st.info("Detected low-energy speech → sadness likely")
            elif emotion == "happy":
                st.success("Expressive tone → happiness likely")
            elif emotion == "angry":
                st.warning("Strong vocal intensity → anger possible")
            else:
                st.write("Balanced speech → neutral")

            # =========================
            # DOWNLOAD REPORT
            # =========================
            st.subheader("📥 Download Report")

            report = {
                "emotion": emotion,
                "confidence": confidence,
                "analysis": analysis
            }

            st.download_button(
                label="Download JSON Report",
                data=json.dumps(report, indent=4),
                file_name="audio_report.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Built by **Kritika Bansal**")