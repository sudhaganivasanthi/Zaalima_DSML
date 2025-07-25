# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('xgboost_fraud_model.pkl')

st.title("💳 Real-Time Fraud Detection")
st.write("Enter transaction details to check if it's fraudulent.")

# Input features
v_features = [f'V{i}' for i in range(1, 29)]

inputs = {}
for v in v_features:
    inputs[v] = st.number_input(v, value=0.0)

hour = st.slider('Hour of Day', 0, 23, 12)
amount = st.number_input('Transaction Amount', min_value=0.0)

# Derived feature
log_amount = np.log1p(amount)

# Assemble into DataFrame
input_data = pd.DataFrame([{
    **inputs,
    'Hour': hour,
    'LogAmount': log_amount
}])

# Predict
if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.write(f"🧠 Prediction: {'FRAUD' if prediction == 1 else 'NOT FRAUD'}")
    st.write(f"📊 Confidence: {prob:.4f}")
