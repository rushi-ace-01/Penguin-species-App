import streamlit as st

st.title('🤖 House Prediction App')

import streamlit as st
import pickle
import numpy as np
# Load the CatBoost model
with open('catboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.write("""
This app predicts house prices based on various features of the property. 
Please input the values for the features below to get a price prediction.
""")

# Create input fields for all the features
MSSubClass = st.number_input('MSSubClass', min_value=0, max_value=200, value=60)
LotFrontage = st.number_input('LotFrontage', min_value=0, max_value=300, value=70)
LotArea = st.number_input('LotArea', min_value=0, max_value=20000, value=7000)
OverallQual = st.number_input('OverallQual', min_value=1, max_value=10, value=5)
OverallCond = st.number_input('OverallCond', min_value=1, max_value=10, value=5)
YearBuilt = st.number_input('YearBuilt', min_value=1800, max_value=2023, value=2000)
YearRemodAdd = st.number_input('YearRemodAdd', min_value=1800, max_value=2023, value=2000)
# MasVnrArea = st.number_input('MasVnrArea', min_value=0, max_value=2000, value=100)
# BsmtFinSF1 = st.number_input('BsmtFinSF1', min_value=0, max_value=2000, value=500)
# BsmtFinSF2 = st.number_input('BsmtFinSF2', min_value=0, max_value=2000, value=0)
# BsmtUnfSF = st.number_input('BsmtUnfSF', min_value=0, max_value=2000, value=500)
# TotalBsmtSF = st.number_input('TotalBsmtSF', min_value=0, max_value=4000, value=1000)
# firstFlrSF = st.number_input('1stFlrSF', min_value=0, max_value=4000, value=1000)
# secondFlrSF = st.number_input('2ndFlrSF', min_value=0, max_value=4000, value=500)
# LowQualFinSF = st.number_input('LowQualFinSF', min_value=0, max_value=2000, value=0)
# GrLivArea = st.number_input('GrLivArea', min_value=0, max_value=5000, value=1500)
# BsmtFullBath = st.number_input('BsmtFullBath', min_value=0, max_value=5, value=1)
# BsmtHalfBath = st.number_input('BsmtHalfBath', min_value=0, max_value=5, value=0)
# FullBath = st.number_input('FullBath', min_value=0, max_value=5, value=2)
# HalfBath = st.number_input('HalfBath', min_value=0, max_value=5, value=1)
# BedroomAbvGr = st.number_input('BedroomAbvGr', min_value=0, max_value=10, value=3)
# KitchenAbvGr = st.number_input('KitchenAbvGr', min_value=0, max_value=5, value=1)
# TotRmsAbvGrd = st.number_input('TotRmsAbvGrd', min_value=0, max_value=15, value=7)
# Fireplaces = st.number_input('Fireplaces', min_value=0, max_value=5, value=1)
# GarageCars = st.number_input('GarageCars', min_value=0, max_value=5, value=2)
# Alloca = st.selectbox('Alloca', [0, 1], index=0)
# Family = st.selectbox('Family', [0, 1], index=0)
# Normal = st.selectbox('Normal', [0, 1], index=1)
# Partial = st.selectbox('Partial', [0, 1], index=0)

# Button for prediction
if st.button('Predict House Price'):
    # Prepare the features as a numpy array
    features = np.array([[MSSubClass, LotFrontage, LotArea, OverallQual, OverallCond, 
                          YearBuilt, YearRemodAdd, MasVnrArea, BsmtFinSF1, BsmtFinSF2, 
                          BsmtUnfSF, TotalBsmtSF, firstFlrSF, secondFlrSF, LowQualFinSF, 
                          GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, 
                          BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, 
                          GarageCars, Alloca, Family, Normal, Partial]])

    # Make a prediction
    prediction = model.predict(features)[0]
    
    # Display the prediction
    st.success(f"The predicted house price is ${prediction:,.2f}")

