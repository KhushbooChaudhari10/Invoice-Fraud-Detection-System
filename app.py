import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler

# Load saved model and preprocessing tools
model = joblib.load('XGBoost_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')
scaler = joblib.load('scaler.joblib')

st.set_page_config(page_title="Invoice Fraud Detection", layout="wide")
st.title("Invoice Fraud Detection System")
st.write("Upload your invoice data below to detect potentially fraudulent records using the trained XGBoost model.")

# File uploader
uploaded_file = st.file_uploader("Upload Invoice CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # --- Data preprocessing ---
    st.write("Preprocessing data...")

    # Ensure date columns are datetime
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
    df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')

    # Derived features
    df['is_duplicate_invoice'] = df['invoice_id'].duplicated(keep=False).astype(int)
    df['invoice_id_length'] = df['invoice_id'].apply(len)
    df['invoice_id_digits'] = df['invoice_id'].str.extract(r'(\d+)').fillna(0).astype(int)
    df['days_to_due'] = (df['due_date'] - df['invoice_date']).dt.days.fillna(0).clip(lower=0)

    # Clean description
    def clean_spaces_only(text):
        if isinstance(text, str):
            return re.sub(r'\s+', ' ', text).strip()
        return ""
    df['cleaned_description'] = df['description'].apply(clean_spaces_only)

    # TF-IDF transformation
    X_text = tfidf.transform(df['cleaned_description'])

    # Numeric features same as training
    X_numeric = df[['amount', 'is_duplicate_invoice', 'invoice_id_length', 'invoice_id_digits', 'days_to_due']]
    X_numeric_scaled = scaler.transform(X_numeric)

    # Combine numeric + text
    X_combined = hstack([X_numeric_scaled, X_text])

    # --- Prediction ---
    st.write("Running fraud detection model...")
    fraud_proba = model.predict_proba(X_combined)[:, 1]
    fraud_pred = (fraud_proba > 0.58).astype(int)  # threshold same as training

    # Add predictions to DataFrame
    df['fraud_probability'] = fraud_proba
    df['fraud_flag'] = fraud_pred

    # Show results
    st.subheader("Detection Results")
    st.write("Flagged invoices (fraudulent or suspicious):")
    flagged = df[df['fraud_flag'] == 1].sort_values("fraud_probability", ascending=False)
    st.dataframe(flagged[['invoice_id', 'vendor_name', 'amount','anomaly_reason', 'fraud_probability']].head(30))

    st.metric("Total Invoices", len(df))
    st.metric("Flagged as Fraudulent", len(flagged))

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇Download Results CSV",
        data=csv,
        file_name="fraud_detection_results.csv",
        mime="text/csv",
    )
else:
    st.info("Please upload a CSV file to begin.")
