import streamlit as st

st.title('🎈 App Name')

st.write('Hello world!')

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# App Title and Description
st.title("❤️ Heart Disease Prediction App")
st.write("This app uses a Logistic Regression model to predict whether a person has heart disease based on their medical parameters.")

# Load Data
@st.cache_data
def load_data():
    # Update the file path as necessary if running locally
    df = pd.read_csv("heart_disease.csv")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Display raw data preview if requested
if st.checkbox("Show Raw Dataset"):
    st.subHeader("Dataset Preview")
    st.dataframe(df.head())

# Preprocessing and Training Model
# Encoding 'Gender' column if it's categorical (e.g., Male/Female)
df_model = df.copy()
le = LabelEncoder()
if 'Gender' in df_model.columns:
    df_model['Gender'] = le.fit_transform(df_model['Gender'])  # Male/Female -> 1/0 (or vice versa)

# Features and Target
X = df_model.drop(columns=['Heart_Disease'])
y = df_model['Heart_Disease']

# Train-Test Split & Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Show model accuracy in sidebar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.sidebar.subheader("Model Performance")
st.sidebar.write(f"**Accuracy:** {accuracy * 100:.2f}%")

# User Input Form in Sidebar / Main Page
st.subheader("Enter Patient Details for Prediction:")

# Creating input widgets dynamically based on features
age = st.slider("Age", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].mean()))

# Handling Gender mapping cleanly
gender_input = st.selectbox("Gender", df['Gender'].unique())
gender = 1 if gender_input == "Male" else 0

resting_bp = st.slider("Resting Blood Pressure", int(df['Resting_Blood_Pressure'].min()), int(df['Resting_Blood_Pressure'].max()), int(df['Resting_Blood_Pressure'].mean()))
cholesterol = st.slider("Cholesterol", int(df['Cholesterol'].min()), int(df['Cholesterol'].max()), int(df['Cholesterol'].mean()))
max_heart_rate = st.slider("Maximum Heart Rate", int(df['Maximum_Heart_Rate'].min()), int(df['Maximum_Heart_Rate'].max()), int(df['Maximum_Heart_Rate'].mean()))

# Prediction Button
if st.button("Predict Heart Disease"):
    # Prepare input array
    input_data = np.array([[age, gender, resting_bp, cholesterol, max_heart_rate]])
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"⚠️ The model predicts that the patient **has a risk of Heart Disease** (Confidence: {prediction_proba[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ The model predicts that the patient **does NOT have a risk of Heart Disease** (Confidence: {prediction_proba[0][0]*100:.2f}%)")
