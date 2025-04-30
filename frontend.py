import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained ML model (AdaBoost)
with open("model.pkl", "rb") as model_file:
    ml_model = pickle.load(model_file)

# Load the trained Deep Learning model
deep_model = tf.keras.models.load_model("deep_model.h5")

# Load the scaler (ideally, load the actual saved scaler; here fitted manually for now)
scaler = StandardScaler()
scaler.fit([[4,146,85,27,100,28.9,0.189,27]])

def predict_diabetes(features, model_type="ML"):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    if model_type == "ML":
        prediction = ml_model.predict(features_scaled)[0]
        return "diabetic" if prediction == 1 else "not diabetic"
    else:
        prediction = deep_model.predict(features_scaled)
        prediction = (prediction > 0.5).astype(int)
        return "diabetic" if prediction[0][0] == 1 else "not diabetic"

# --- Streamlit Frontend ---
st.set_page_config(page_title="Diabetes Prediction", page_icon="🌿", layout="centered")

st.title("Diabetes Prediction Model")
st.markdown("""
Welcome to the **Diabetes Prediction App using Machine Learning and Deep Learning**. Fill out the following details, choose your prediction model, and find out your diabetes risk.
""")

st.sidebar.header("Select Model")
model_choice = st.sidebar.radio("Choose prediction model:", ("Machine Learning (AdaBoost)", "Deep Learning (Neural Network)"))

st.subheader("Patient Medical Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    model_type = "ML" if model_choice == "Machine Learning (AdaBoost)" else "DL"
    prediction = predict_diabetes(features, model_type)

    st.subheader("Prediction Result")
    if prediction.lower() == "diabetic":
        st.error("🚨 High risk of diabetes detected!")

    else:
        st.success("\u2705 Low risk of diabetes.")

st.markdown("---")

