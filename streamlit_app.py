
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
try:
    with open('logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'logistic_regression_model.pkl' not found. Please ensure the model is trained and saved.")
    st.stop()

st.title('Plant Irrigation Prediction App')
st.write('Enter the soil moisture level to predict if the pump should be ON or OFF.')

# Input for Soil Moisture
soil_moisture_input = st.slider(
    'Soil Moisture (%)',
    min_value=0.0,
    max_value=100.0,
    value=25.0,
    step=0.1
)

# Prepare input for the model
# The model was trained with only 'Soil_Moisture'
input_df = pd.DataFrame({'Soil_Moisture': [soil_moisture_input]})

if st.button('Predict Pump Status'):
    if model is not None:
        prediction = model.predict(input_df)
        status = "ON" if prediction[0] == 1 else "OFF"

        st.subheader('Prediction Result:')
        if status == "ON":
            st.success(f"The pump should be: **{status}**")
        else:
            st.info(f"The pump should be: **{status}**")
    else:
        st.warning("Model is not loaded. Cannot make predictions.")
