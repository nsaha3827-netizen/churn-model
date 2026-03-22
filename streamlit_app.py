import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
# ---------------------------
# SAFE LOAD FUNCTION
# ---------------------------
def safe_load(path, key=None):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj.get(key, obj)
    return obj

# ---------------------------
# LOAD FILES SAFELY
# ---------------------------
try:
    model = safe_load("model.pkl", "model")
    pipeline = safe_load("pipeline.pkl", "pipeline")
    columns_info = joblib.load("columns.pkl")

    num_cols = columns_info["num"]
    cat_cols = columns_info["cat"]

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ---------------------------
# FIX SeniorCitizen
# ---------------------------
if "SeniorCitizen" in num_cols:
    num_cols.remove("SeniorCitizen")
    cat_cols.append("SeniorCitizen")

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Churn AI", layout="wide")

st.title("🚀 Customer Churn Prediction")
st.markdown("AI-powered customer retention system")

input_data = {}

# ---------------------------
# NUMERIC INPUTS
# ---------------------------
st.subheader("Customer Metrics")

for col in num_cols:
    input_data[col] = st.number_input(col, value=0.0)

# ---------------------------
# DROPDOWN OPTIONS
# ---------------------------
dropdown_options = {
    "SeniorCitizen": ["Yes", "No"],
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

# ---------------------------
# CATEGORICAL INPUTS
# ---------------------------
st.subheader("Customer Profile")

for col in cat_cols:
    if col in dropdown_options:
        input_data[col] = st.selectbox(col, dropdown_options[col])
    else:
        input_data[col] = st.text_input(col)

# ---------------------------
# CONVERT VALUES
# ---------------------------
if "SeniorCitizen" in input_data:
    input_data["SeniorCitizen"] = 1 if input_data["SeniorCitizen"] == "Yes" else 0

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("Analyze Customer"):

    try:
        df = pd.DataFrame([input_data])

        # Convert numeric safely
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check missing values
        if df.isnull().sum().sum() > 0:
            st.warning("⚠️ Some fields are missing or invalid")
            st.stop()

        # Apply pipeline
        X = pipeline.transform(df)

        # Predict
        prob = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]

        percentage = round(prob * 100, 2)
        

# ------------------ CHART ------------------
        stay = round(100 - percentage, 2)

        st.subheader("📊 Churn Probability Distribution")
        

        # Convert to 0–1 for progress bar
        progress_value = percentage / 100

        st.progress(progress_value)

        st.write(f"**Risk Score: {percentage}%**")
        if percentage > 60:
            st.error(f"⚠️ High Risk of Churn: {percentage}%")
        elif percentage > 30:
            st.warning(f"⚠️ Moderate Risk: {percentage}%")
        else:
            st.success(f"✅ Low Risk (Customer likely to stay): {100 - percentage}%")

            st.subheader("📊 Prediction Result")

            

        st.progress(int(percentage))

    except Exception as e:
        st.error(f"Prediction error: {e}")