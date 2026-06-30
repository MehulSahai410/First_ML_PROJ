import streamlit as st
import pickle
import numpy as np


model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Student Placement Predictor")
st.write("Enter the student's details below to predict placement chances.")


cgpa = st.number_input("Enter CGPA (e.g., 7.5)", min_value=0.0, max_value=10.0, step=0.1)
iq = st.number_input("Enter IQ (e.g., 110)", min_value=0, max_value=250, step=1)


if st.button("Predict"):
    
    features = np.array([[cgpa, iq]])
    
    
    features_scaled = scaler.transform(features)
    
    
    prediction = model.predict(features_scaled)
    
    
    if prediction[0] == 1:
        st.success("Result: Ho jaega placement")
    else:
        st.error("Result: Nhi hoega placement")