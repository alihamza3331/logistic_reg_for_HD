import streamlit as st

st.title('🎈 App Name')

st.write('Hello world!')

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# Load the trained model and scaler
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("logistic_reg (1).joblib")
    return scaler, model

try:
    scaler, model = load_artifacts()
except Exception as e:
    st.error(f"Error loading model or scaler files: {e}")
    st.info("Ensure 'scaler.joblib' and 'logistic_reg (1).joblib' are present in the same folder as app.py.")
    st.stop()

# Header
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient clinical parameters below to evaluate heart disease risk.")

# Optional: Expandable section to inspect expected model features
with st.expander("🔍 Debug Info: Expected Model Features"):
    if hasattr(scaler, "feature_names_in_"):
        st.write("**Scaler expects these features in order:**", list(scaler.feature_names_in_))
    else:
        st.write("Scaler was trained without feature names.")

# Input Form
with st.form("prediction_form"):
    st.subheader("Patient Attributes")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120, step=1)

    with col2:
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=220, step=1)
        max_hr = st.number_input("Maximum Heart Rate", min_value=50, max_value=250, value=150, step=1)

    submit = st.form_submit_button("Predict Risk")

if submit:
    try:
        # Map Gender string to numerical value
        gender_encoded = 1 if gender == "Male" else 0

        # Build dictionary with input values
        raw_inputs = {
            'age': age,
            'gender': gender_encoded,
            'sex': gender_encoded,
            'resting_blood_pressure': resting_bp,
            'trestbps': resting_bp,
            'cholesterol': cholesterol,
            'chol': cholesterol,
            'maximum_heart_rate': max_hr,
            'thalach': max_hr
        }

        # Align inputs dynamically with expected scaler feature names
        if hasattr(scaler, "feature_names_in_"):
            expected_cols = list(scaler.feature_names_in_)
            input_dict = {}

            for col in expected_cols:
                col_clean = col.lower().replace(' ', '_')
                
                # Match feature names based on keywords
                if 'age' in col_clean:
                    input_dict[col] = age
                elif 'gen' in col_clean or 'sex' in col_clean:
                    input_dict[col] = gender_encoded
                elif 'bp' in col_clean or 'press' in col_clean or 'trest' in col_clean:
                    input_dict[col] = resting_bp
                elif 'chol' in col_clean:
                    input_dict[col] = cholesterol
                elif 'hr' in col_clean or 'heart' in col_clean or 'thalach' in col_clean:
                    input_dict[col] = max_hr
                else:
                    input_dict[col] = 0

            input_data = pd.DataFrame([input_dict])[expected_cols]
        else:
            # Fallback default order
            input_data = pd.DataFrame([{
                'Age': age,
                'Gender': gender_encoded,
                'Resting_Blood_Pressure': resting_bp,
                'Cholesterol': cholesterol,
                'Maximum_Heart_Rate': max_hr
            }])

        # Scale input features
        scaled_input = scaler.transform(input_data)

        # Predict class and probability
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ **High Risk of Heart Disease** (Probability: {prediction_proba:.1%})")
            st.write("The model indicates an elevated likelihood of heart disease. Clinical follow-up is recommended.")
        else:
            st.success(f"✅ **Low Risk of Heart Disease** (Probability: {prediction_proba:.1%})")
            st.write("The model indicates a low likelihood of heart disease based on the provided inputs.")

    except Exception as err:
        st.error(f"Error performing prediction: {err}")
