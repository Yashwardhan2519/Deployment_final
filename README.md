Project Overview
This project leverages machine learning to predict whether a customer will churn (leave a service) based on their service usage and demographic attributes. It uses an XGBoost classification model trained on the Telco Customer Churn dataset.
The goal is to identify high-risk customers proactively and support business strategies focused on retention, loyalty, and customer lifetime value enhancement.

Business Problem
Customer churn is a critical metric for subscription-based businesses like telecom, streaming services, and SaaS. Losing a customer not only affects immediate revenue but also the long-term growth trajectory of the company.
By predicting churn likelihood, companies can:

Reduce customer attrition

Personalize retention campaigns

Improve resource allocation for customer success teams

Boost profitability

Customer_Service_Prediction/
│
├── deployment.py            # Streamlit web application for real-time prediction
├── model.pkl                # Pre-trained XGBoost model
├── requirements.txt         # Dependencies to run the project
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Original dataset
└── README.md                # This documentation file

Model
Algorithm: XGBoost Classifier

Scaler: MinMaxScaler for numeric normalization

Encoding: One-Hot Encoding for categorical variables

The model has been evaluated using:

Confusion Matrix

Accuracy Score

Classification Report

Streamlit Web App
The deployment.py file creates a simple, interactive web application that allows users to input customer data and get instant churn predictions.


This tool is a prototype and should be tested in a controlled environment before deployment in production. Model performance can vary based on data freshness and market dynamics.

