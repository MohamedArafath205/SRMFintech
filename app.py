import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv("fintech3.csv")
data['Properties'].fillna(0, inplace=True)
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Insurance'] = data['Insurance'].map({'Yes': 1, 'No': 0})
data['Demographic'] = data['Demographic'].map({'Rural': 0, 'Suburban': 1, 'Urban': 2})
data['Marital_status'] = data['Marital_status'].map({'Married': 1, 'Single': 0})
data['Properties'] = data['Properties'].map({'Apartment': 1, 'Condo': 2, 'House': 3})
data['Emp_status'] = data['Emp_status'].map({'Self Employed': 1, 'Employee': 2, 'Entrepreneur': 3})
data['Properties'].fillna(0, inplace=True)

X = data.drop("Fin_Cat", axis=1)
y = data["Fin_Cat"]

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train logistic regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = logreg_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy)

# Streamlit interface
st.title("Credit Score Improvement Model")
st.write("""
Your credit score acts as a pivotal metric that can significantly impact your ability to access various opportunities and services. 
A robust credit score is more than just a number; it's a reflection of your financial responsibility and trustworthiness in the eyes of lenders, 
landlords, and even potential employers. A high credit score opens doors to favorable interest rates on loans, increased chances of loan approval, 
lower insurance premiums, and even better rental or employment prospects. Conversely, a low credit score can limit your financial options, leading 
to higher interest rates, restricted access to credit, and potential barriers to securing housing or employment. Therefore, understanding and actively 
managing your credit score is essential for navigating the complexities of today's financial landscape and achieving your long-term financial goals.
""")

st.sidebar.title("Enter your details")

# Collect user inputs
age = st.sidebar.number_input('Age', step=1)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
annual_income = st.sidebar.number_input('Annual Income', step=1000)
loan_amount = st.sidebar.number_input('Loan Amount', step=1000)
savings_amount = st.sidebar.number_input('Savings Amount', step=100)
investment_amount = st.sidebar.number_input('Investment Amount', step=100)
credit_score = st.sidebar.number_input('Credit Score', step=1)
insurance = st.sidebar.selectbox('Insurance', ['Yes', 'No'])
demographic = st.sidebar.selectbox('Demographic', ['Rural', 'Suburban', 'Urban'])
dependents = st.sidebar.number_input('Number of Dependents', step=1)
education_level = st.sidebar.number_input('Education Level', step=1)
marital_status = st.sidebar.selectbox('Marital Status', ['Married', 'Single'])
properties = st.sidebar.selectbox('Properties', ['Apartment', 'Condo', 'House'])
emp_status = st.sidebar.selectbox('Employment Status', ['Self Employed', 'Employee', 'Entrepreneur'])

# Map user inputs to numerical values
input_data = np.array([
    age,
    1 if gender == 'Male' else 0,
    annual_income,
    loan_amount,
    savings_amount,
    investment_amount,
    credit_score,
    1 if insurance == 'Yes' else 0,
    {'Rural': 0, 'Suburban': 1, 'Urban': 2}[demographic],
    dependents,
    education_level,
    1 if marital_status == 'Married' else 0,
    {'Apartment': 1, 'Condo': 2, 'House': 3}[properties],
    {'Self Employed': 1, 'Employee': 2, 'Entrepreneur': 3}[emp_status]
]).reshape(1, -1)

input_data_scaled = scaler.transform(input_data)
predicted_outcome_logreg = logreg_model.predict(input_data_scaled)

def display_insights(predicted_label):
    if predicted_label == 0:
        st.write("🛑 *High Risk Level* 🛑")
        st.write("* Start by checking your credit report regularly to identify any errors or discrepancies that may be affecting your score.")
        st.write("* Focus on making timely payments for all your bills, including credit cards, loans, and utility bills.")
        st.write("* Keep your credit card balances low and aim to pay off outstanding balances in full each month to avoid high-interest charges.")
        st.write("* Consider applying for a secured credit card to build credit history if you have limited or no credit history.")
        st.write("* Avoid opening multiple new credit accounts within a short period, as this can lower your average account age and impact your score negatively.")
        st.write("* Avoid availing unnecessary loans and consider applying for an insurance.")
    elif predicted_label == 1:
        st.write("⚠ *Moderate Level* ⚠")
        st.write("* Take advantage of credit monitoring services offered by credit bureaus to stay informed about changes to your credit report.")
        st.write("* Set up automatic payments for your bills to ensure you never miss a payment deadline.")
        st.write("* Aim to reduce your overall debt-to-income ratio by paying down existing debts and avoiding taking on new debt unnecessarily.")
        st.write("* If you have any past-due accounts, work with creditors to negotiate payment plans and bring them current.")
        st.write("* Consider becoming an authorized user on a family member's credit card with a positive payment history to help boost your credit score.")
        st.write("* Think wisely before applying for future loans.")
    else:
        st.write("✅ *Good Going!* ✅")
        st.write("* Consider applying for a credit builder loan or a secured loan to demonstrate responsible borrowing behavior and improve your credit mix.")

if st.sidebar.button("Predict Credit Score Level"):
    display_insights(predicted_outcome_logreg[0])

st.write("## Feature Significance")

input_weights = logreg_model.coef_[0]
feature_names = X.columns
indices = np.argsort(np.abs(input_weights))[::-1]

plt.figure(figsize=(12, 8))
plt.bar(range(len(input_weights)), np.abs(input_weights[indices]), align="center")
plt.xticks(range(len(input_weights)), feature_names[indices], rotation=45)
plt.title("Feature Importances - Logistic Regression Model")
st.pyplot(plt)
