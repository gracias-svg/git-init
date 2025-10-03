import streamlit as st
import pandas as pd
import joblib

st.title("Customer Health & Churn Prediction")

model = joblib.load('lightgbm_model.joblib')  # or 'logistic_regression_model.joblib'
uploaded = st.file_uploader("Upload customers CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    
    # Preprocess: Convert categorical columns to numeric
    # Drop customerID (not a feature) and Churn (target column)
    df_processed = df.copy()
    if 'customerID' in df_processed.columns:
        df_processed = df_processed.drop('customerID', axis=1)
    if 'Churn' in df_processed.columns:
        df_processed = df_processed.drop('Churn', axis=1)
    
    # Convert TotalCharges to numeric (it may be object type)
    if 'TotalCharges' in df_processed.columns:
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce').fillna(0)
    
    # One-hot encode all object/categorical columns
    df_processed = pd.get_dummies(df_processed, drop_first=True)
    
    # Make predictions
    preds = model.predict_proba(df_processed)[:, 1]
    df['churn_prob'] = preds
    df['health_score'] = 1 - df['churn_prob']
    st.dataframe(df[['churn_prob', 'health_score']].head())
    st.download_button("Download scored CSV", df.to_csv(index=False), file_name="scored.csv")
