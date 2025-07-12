import joblib
import streamlit as st
import pandas as pd

# Add imports for classes used inside pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


st.set_page_config(page_title="Loan Approval - Random Forest", page_icon="🏦")
st.title("🏦 Loan Approval Predictor (Random Forest)")

@st.cache_resource
def load_model():
    return joblib.load("rf_loan_pipeline.pkl")

pipeline = load_model()

# Inputs
gender       = st.selectbox("Gender", ["Male", "Female"])
married      = st.selectbox("Married", ["Yes", "No"])
education    = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed= st.selectbox("Self Employed", ["Yes", "No"])
property_area= st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

col1, col2 = st.columns(2)
with col1:
    applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
    loan_amount      = st.number_input("Loan Amount (₹ thousand)", min_value=0, step=1)
with col2:
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
    loan_term         = st.number_input("Loan Amount Term (months)", min_value=12, step=12)

credit_history = st.selectbox("Credit History", ["Meets credit criteria (1)", "Does NOT meet (0)"])
credit_hist_val = 1 if credit_history.startswith("Meets") else 0

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_hist_val,
        "Gender": gender,
        "Married": married,
        "Education": education,
        "Self_Employed": self_employed,
        "Property_Area": property_area
    }])

    prediction = pipeline.predict(input_df)[0]
    prediction_proba = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Loan likely approved with confidence ≈ {prediction_proba:.0%}")
    else:
        st.error(f"❌ Loan likely rejected with confidence ≈ {1 - prediction_proba:.0%}")
