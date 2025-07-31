# الكود بالكامل مع تعديل استخدام موديل واحد فقط اسمه final_model.pkl
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ========== CONFIG ==========
st.set_page_config(page_title="Cyber Attack Detector", layout="wide")

st.markdown("""
<style>
body { background-color: #121212; color: #f0f0f0; }
.main { background-color: #1e1e1e; }
h1, h2, h3 { color: gold; }
.stButton>button { background-color: #333333; color: gold; border-radius: 10px; border: 1px solid gold; }
</style>
""", unsafe_allow_html=True)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    model = joblib.load("model/final_model.pkl")
    return model

model = load_model()

# ========== FEATURE LIST ==========
required_columns = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'sload', 'dload']

# ========== PREDICTION FUNCTION ==========
def make_prediction(df):
    model_input = df[required_columns].astype(float)
    prediction = model.predict(model_input)
    df['Prediction'] = prediction
    return df

# ========== SIDEBAR INPUT ==========
st.sidebar.title("Upload or Input Data")
data_source = st.sidebar.radio("Choose input method:", ["Upload .parquet file", "Manual input"])

input_df = pd.DataFrame()

if data_source == "Upload .parquet file":
    uploaded_file = st.sidebar.file_uploader("Upload Parquet File", type=["parquet"])
    if uploaded_file:
        input_df = pd.read_parquet(uploaded_file)
        st.success("File uploaded successfully.")
else:
    st.sidebar.write("Enter values for the following fields:")
    row_data = {}
    for col in required_columns:
        val = st.sidebar.text_input(f"{col}")
        row_data[col] = val

    if all(v != '' for v in row_data.values()):
        input_df = pd.DataFrame([row_data])
    else:
        input_df = pd.DataFrame()

# ========== MAIN AREA ==========
st.title("🚨 Cyber Attack Detection Dashboard")

if not input_df.empty:
    st.subheader("📊 Uploaded / Entered Data Preview")
    st.dataframe(input_df.head())

if st.button("Predict"):
    with st.spinner("Analyzing data..."):
        prediction_df = make_prediction(input_df.copy())

        st.subheader("🔍 Prediction Results")
        st.dataframe(prediction_df)

        attack_counts = prediction_df['Prediction'].value_counts()

        st.markdown("---")
        st.subheader("📈 Attack Statistics")
        st.write("### Attack Count:")
        st.write(attack_counts)

        # Plot pie chart
        fig1, ax1 = plt.subplots()
        ax1.pie(attack_counts, labels=attack_counts.index, autopct='%1.1f%%', colors=sns.color_palette('dark'))
        ax1.set_title("Attack Distribution")
        st.pyplot(fig1)

        # Plot bar chart
        fig2, ax2 = plt.subplots()
        sns.barplot(x=attack_counts.index, y=attack_counts.values, palette="rocket", ax=ax2)
        ax2.set_title("Number of Attacks by Type")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

        # Optional CSV download
        csv = prediction_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")
else:
    st.info("Awaiting data input...")
