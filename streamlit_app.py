import streamlit as st
import logging
import numpy as np
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

try:
    with open('catboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    st.error("Failed to load the model.")

try:
    # Replace this part with your input fields
    MSSubClass = st.number_input('MSSubClass', min_value=0, max_value=200, value=60)
    # Other inputs...
    
    # Button for prediction
    if st.button('Predict House Price'):
        features = np.array([[MSSubClass, LotFrontage, LotArea, ...]])  # Fill in all input fields
        prediction = model.predict(features)[0]
        st.success(f"The predicted house price is ${prediction:,.2f}")
except ValueError as ve:
    logger.error(f"ValueError: {ve}")
    st.error(f"ValueError: {ve}")
except Exception as e:
    logger.error(f"Error: {e}")
    st.error(f"Error: {e}")
