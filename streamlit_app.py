import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Title and Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🐧 Penguin Species Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #0F9D58;'>Predict the species of a penguin using machine learning</h3>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header('Input Features')
    island = st.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
    sex = st.selectbox('Sex', ('male', 'female'))
    bill_length_mm = st.slider('Bill length (mm)', 39.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.image("https://static.streamlit.io/examples/penguin.jpg", width=300)

# Input Data
data = {'island': island,
        'sex': sex,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'body_mass_g': body_mass_g,
        'flipper_length_mm': flipper_length_mm}
input_df = pd.DataFrame(data, index=[0])

# Load Data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
X_raw = df.drop('species', axis=1)
y_raw = df['species']

# Combine Input Data with the dataset for encoding
input_penguins = pd.concat([input_df, X_raw], axis=0)
df_penguins = pd.get_dummies(input_penguins, columns=['island', 'sex'])

X = df_penguins[1:]  # Extract input features for the entire dataset
input_row = df_penguins[:1]  # Extract only the input row for prediction

# Target Encoding
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
y = y_raw.map(target_mapper)

# Model Training
clf = RandomForestClassifier()
clf.fit(X, y)

# Make Predictions
prediction = clf.predict(input_row)
prediction_prob = clf.predict_proba(input_row)

df_prediction_prob = pd.DataFrame(prediction_prob, columns=['Adelie', 'Chinstrap', 'Gentoo'])

# Output Section
st.subheader('Prediction Probability')
st.bar_chart(df_prediction_prob.T)

# Show the final prediction
st.markdown(f"<h2 style='text-align: center; color: #FF6347;'>Prediction: {penguins_species[prediction][0]}</h2>", unsafe_allow_html=True)

# Add a success message
st.success(f'The predicted species is **{penguins_species[prediction][0]}**')
