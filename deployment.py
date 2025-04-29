import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import pickle

# Load and clean data
df = pd.read_csv(r'C:\Users\Yash\OneDrive\Desktop\GitHub_Repositorites\Customer_Service_prediction\Deployment_final\WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.columns = df.columns.str.replace(' ', '_')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop(columns='customerID', inplace=True)

# One-hot encode
df = pd.get_dummies(df)
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_scaled, y)

# Save all components in one file
model_bundle = {
    "model": model,
    "scaler": scaler,
    "columns": X.columns.tolist()
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)
