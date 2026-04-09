# 🎙️ Acoustic Emotion Analysis using Machine Learning

> 🚀 An intelligent system that understands **how you speak**, not just **what you say**.

---

## 🌟 Overview

This project is a **real-time Audio Behavior Analysis System** that detects human emotions from speech using **Machine Learning and acoustic signal processing**.

Unlike traditional classifiers, this system goes beyond prediction — it provides **insights into speech patterns** like energy, pitch, and pauses to make results more **interpretable and meaningful**.

---

## 🎯 What Makes This Project Unique?

* 🧠 **Interpretable AI** – Not just predictions, but reasoning using audio signals
* 🎤 **Real-time Voice Input** – Analyze emotions directly from microphone
* 📊 **Acoustic Insights** – Understand *why* a prediction was made
* ⚡ **No Retraining Required** – Pre-trained model included
* 💡 **Clean UI with Streamlit** – Simple yet powerful interface

---

## 🚀 Features

* 🎯 Emotion Detection: Angry 😠 | Happy 😊 | Sad 😢 | Neutral 😐
* 📈 Confidence Score Visualization
* 📊 Audio Feature Analysis:

  * Energy (loudness)
  * Pitch (frequency)
  * Silence Ratio (pauses)
  * Duration
* 🧠 Smart Interpretation of results
* 🎤 Live microphone input
* 📂 WAV file upload
* 📥 Downloadable JSON report

---

## 🧠 Tech Stack

* **Python**
* **Streamlit**
* **Scikit-learn**
* **Librosa (Audio Processing)**
* **NumPy**

---

## 📂 Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

* 🎭 Emotions:

  * Angry 😠
  * Happy 😊
  * Sad 😢
  * Neutral 😐

* 👥 24 professional actors

* 🎧 High-quality WAV recordings

🔗 Dataset Link:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

---

### ⚠️ Note

* The dataset is **not included** in this repository due to size limitations.
* The model is already trained and saved.

👉 **You can run the project directly without retraining.**

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/kritika038/Acoustic-Emotion-Analysis-using-Machine-Learning.git
cd Acoustic-Emotion-Analysis-using-Machine-Learning
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the application

```bash
streamlit run streamlit_app.py
```

---

## 🧪 How It Works

```text
Audio Input → Feature Extraction → Scaling → ML Model → Emotion Prediction → Insights
```

### 🔍 Feature Extraction:

* MFCC (captures speech characteristics)
* Energy (loudness level)
* Pitch (frequency variation)
* Zero Crossing Rate (signal change rate)

### 🤖 Model:

* XGBoost Classifier
* Optimized for small structured audio datasets

---

## 📊 Example Output

* 🎯 Emotion: **Angry 😠**
* 📈 Confidence: **0.95**
* 📊 Energy: High
* 🎵 Pitch: Elevated
* 🔇 Silence: Moderate

👉 Interpretation:
**Strong vocal intensity indicates possible anger**

---

## 📸 Screenshots

📌 *Add screenshots here after deployment*

---

## 💡 Key Highlights

* ✅ End-to-end ML pipeline
* ✅ Real-time inference
* ✅ Feature-driven interpretation
* ✅ Lightweight and deployable
* ✅ Clean and user-friendly UI

---

## 🔮 Future Improvements

* 🎙️ Continuous real-time emotion tracking
* 📊 Advanced visualizations (waveforms, spectrograms)
* 🤖 Deep Learning model integration (CNN/RNN)
* 🌐 API deployment for external integration

---

## 👩‍💻 Author

**Kritika**

---

## ⭐ Support

If you found this project useful:

👉 Give it a ⭐ on GitHub
👉 Share it with others

---

> 💬 “Machines can now understand not just words, but emotions behind them.”
