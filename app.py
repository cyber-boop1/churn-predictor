import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from train_model import train_and_save_model

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="centered"
)

# Train model if not already saved
if not os.path.exists("churn_model.pkl"):
    train_and_save_model()

# Load model and scaler
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("churn_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------- UI ----------
st.title("🔮 Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to leave — built with Machine Learning!")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("📅 Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("💵 Monthly Charges (₹)", 500, 10000, 3000)
    total_charges = st.number_input("💰 Total Charges (₹)", min_value=0, value=tenure * monthly_charges)

    contract = st.selectbox("📄 Contract Type", [
        "Month-to-month", "One year", "Two year"
    ])

with col2:
    internet_service = st.selectbox("🌐 Internet Service", [
        "DSL", "Fiber optic", "No"
    ])
    payment_method = st.selectbox("💳 Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer", "Credit card"
    ])
    tech_support = st.selectbox("🛠 Tech Support", ["Yes", "No"])
    senior_citizen = st.checkbox("👴 Senior Citizen")

st.markdown("---")

if st.button("🔮 Predict Churn Risk", use_container_width=True):

    # Encode inputs
    contract_map  = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map  = {"DSL": 0, "Fiber optic": 1, "No": 2}
    payment_map   = {"Electronic check": 0, "Mailed check": 1,
                     "Bank transfer": 2, "Credit card": 3}
    support_map   = {"Yes": 1, "No": 0}

    input_data = np.array([[
        tenure,
        monthly_charges,
        total_charges,
        contract_map[contract],
        internet_map[internet_service],
        payment_map[payment_method],
        support_map[tech_support],
        int(senior_citizen)
    ]])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        st.error(f"### ⚠️ High Churn Risk! ({probability:.1f}% probability)")
        st.markdown("#### 💡 Recommended Actions:")
        st.markdown("""
        - 📞 Reach out to customer immediately
        - 🎁 Offer a loyalty discount or upgrade
        - 📄 Suggest switching to a longer contract
        - 🛠 Provide dedicated tech support
        """)
    else:
        st.success(f"### ✅ Low Churn Risk ({probability:.1f}% probability)")
        st.markdown("#### 💡 Retention Tips:")
        st.markdown("""
        - 🌟 Keep engagement high with offers
        - 📊 Monitor usage patterns regularly
        - 🎉 Reward loyalty with perks
        """)

    # Gauge chart
    st.markdown("#### 📊 Churn Risk Meter")
    col3, col4, col5 = st.columns(3)
    col3.metric("Churn Probability", f"{probability:.1f}%")
    col4.metric("Tenure", f"{tenure} months")
    col5.metric("Monthly Charges", f"₹{monthly_charges:,}")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Built by Shalini Jaiswal · "
    "<a href='https://github.com/cyber-boop1'>GitHub</a> · "
    "<a href='https://linkedin.com/in/jaiswal-shalini'>LinkedIn</a></div>",
    unsafe_allow_html=True
)
