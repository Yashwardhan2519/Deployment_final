import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
churn_data= pd.read_csv(r"C:\Users\Yash\OneDrive\Desktop\Python\Customer_Churn_telcom\tel_churn.csv")
churn_data.drop(columns='Unnamed: 0', inplace=True)
X = churn_data.drop('Churn', axis=1)
y = churn_data['Churn']
x_train , x_test , y_train , y_test = train_test_split(X , y , test_size=0.3)
sub_vars = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes','Dependents_No', 'Dependents_Yes', 'PhoneService_No',
'PhoneService_Yes', 'MultipleLines_No',
'MultipleLines_No phone service', 'MultipleLines_Yes',
'InternetService_DSL', 'InternetService_Fiber optic',
'InternetService_No', 'OnlineSecurity_No',
'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
'OnlineBackup_No', 'OnlineBackup_No internet service',
'OnlineBackup_Yes', 'DeviceProtection_No',
'DeviceProtection_No internet service', 'DeviceProtection_Yes',
'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
'StreamingMovies_No', 'StreamingMovies_No internet service',
'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
'PaymentMethod_Bank transfer (automatic)',
'PaymentMethod_Credit card (automatic)',
'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36',
'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform it
x_train[sub_vars] = scaler.fit_transform(x_train[sub_vars])

# Use the same scaler to transform the test data
x_test[sub_vars] = scaler.transform(x_test[sub_vars])

lr = LogisticRegression()
lr.fit(x_train, y_train)


#Intercept and coefficients
print(lr.coef_, lr.intercept_)

y_pred = lr.predict(x_test)

results = confusion_matrix(y_test, y_pred)

print ("Confusion Matrix of test data:")
print(results)

#Another method to calculate accuracy
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(lr.score(x_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(lr.score(x_test, y_test)))

import joblib

# Save the model
joblib.dump(lr, 'logistic_model.pkl')

print("Model saved as logistic_model.pkl")