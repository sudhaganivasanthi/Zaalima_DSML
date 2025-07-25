# app.py
from utils import preprocess, extract_text_from_file
import numpy as np

import streamlit as st
import joblib
from utils import preprocess

# Load model/vectorizer
model = joblib.load("model/classification_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

st.title("🤖 AI Resume Screening System")
st.write("Paste your resume content below and click classify:")

upload_option = st.radio("Choose Input Type:", ["Paste Text", "Upload File"])

if upload_option == "Paste Text":
    text_input = st.text_area("Paste Resume Text Here", height=300)
elif upload_option == "Upload File":
    uploaded_file = st.file_uploader("Upload a resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
    if uploaded_file:
        text_input = extract_text_from_file(uploaded_file)
    else:
        text_input = ""

if st.button("Classify"):
    if text_input.strip():
        cleaned = preprocess(text_input)
        vec = vectorizer.transform([cleaned])
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(vec)[0]
            top_n = sorted(zip(model.classes_, probas), key=lambda x: -x[1])[:3]
            for role, score in top_n:
                st.success(f"🎯 {role} — {score*100:.2f}% confidence")
        else:
            prediction = model.predict(vec)[0]
            st.success(f"🎯 Predicted Job Role: **{prediction}**")
    else:
        st.warning("Please provide resume content.")
