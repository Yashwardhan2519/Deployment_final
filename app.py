import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model bundle
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"], bundle["columns"]

model, scaler, columns = load_model()

st.title("Customer Churn Prediction")

# Collect user input
def get_user_input():
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure", 0, 72, 1)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=2500.0)

    user_dict = {
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner, "Dependents": Dependents,
        "tenure": tenure, "PhoneService": PhoneService, "MultipleLines": MultipleLines,
        "InternetService": InternetService, "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport, "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies, "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges
    }

    return pd.DataFrame([user_dict])

input_df = get_user_input()

# Preprocess input
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_encoded)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    st.write("Prediction: ", "Churn" if prediction == 1 else "No Churn")
