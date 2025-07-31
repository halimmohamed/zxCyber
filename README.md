# 🛡️ Network Attack Detection App

This project is a full web application using Streamlit to detect network intrusions based on the UNSW-NB15 dataset.

## 📂 Project Structure

project/
├── model/
│ └── model.pkl ← Pre-trained ML model
├── data/
│ ├── normal_sample.parquet ← Test sample with normal traffic
│ └── attack_sample.parquet ← Test sample with an attack
├── app/
│ └── app.py ← Streamlit frontend
├── requirements.txt ← Required dependencies
└── README.md ← This file

## 🚀 How to Run the App

### 1. Install requirements

pip install -r requirements.txt

### 2. Launch the Streamlit app

streamlit run app/app.py

🧠 Model Info

The ML model is a binary classifier that detects whether a given row is normal traffic or a malicious attack based on network features.

Trained using SVM or Random Forest depending on the backend logic.

## ⚙️ Features

- Upload .parquet file or enter single row manually
- Visual stats: number and type of attacks
- Friendly interface: shows warning or normal label with custom messages
- Supports bar/pie chart visualizations
- Styled with dark rocky theme and golden accents ✨

## 🔗 GitHub Repository

> [Add GitHub repo link here]

---

Created with by Abdelhalim for cyber security detection project
