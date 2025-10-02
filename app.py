import streamlit as st
import pandas as pd
import joblib

st.title("Customer Health & Churn Prediction")

model = joblib.load('lightgbm_model.joblib')  # or 'logistic_regression_model.joblib'
uploaded = st.file_uploader("Upload customers CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    # Make sure the features column order matches your training!
    preds = model.predict_proba(df)[:, 1]
    df['churn_prob'] = preds
    df['health_score'] = 1 - df['churn_prob']
    st.dataframe(df[['churn_prob', 'health_score']].head())
    st.download_button("Download scored CSV", df.to_csv(index=False), file_name="scored.csv")
