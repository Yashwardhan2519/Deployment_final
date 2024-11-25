from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('logistic_model.pkl')

# Define the feature names
feature_names = [
    'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender_Female', 'gender_Male',
    'Partner_No', 'Partner_Yes', 'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
    'PhoneService_Yes', 'MultipleLines_No', 'MultipleLines_No phone service', 
    'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic',
    'InternetService_No', 'OnlineSecurity_No', 'OnlineSecurity_No internet service', 
    'OnlineSecurity_Yes', 'OnlineBackup_No', 'OnlineBackup_No internet service', 
    'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_No internet service', 
    'DeviceProtection_Yes', 'TechSupport_No', 'TechSupport_No internet service', 
    'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_No internet service', 
    'StreamingTV_Yes', 'StreamingMovies_No', 'StreamingMovies_No internet service', 
    'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year', 
    'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes', 
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 
    'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36', 
    'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        input_data = [float(request.form.get(feature, 0)) for feature in feature_names]
        features = np.array(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)

        # Display result
        return f"""
        <h2>Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}</h2>
        <p>Probability of Churn: {probability[0][1]:.2f}</p>
        <p>Probability of No Churn: {probability[0][0]:.2f}</p>
        <a href="/">Go back</a>
        """
    except Exception as e:
        return f"<h2>Error: {str(e)}</h2>"

if __name__ == '__main__':
    app.run(debug=True)
    
