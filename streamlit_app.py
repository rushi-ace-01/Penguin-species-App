import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This app predicts outcomes based on machine learning model!')

with st.expander('Data'):
  st.write('**Raw Data**')
  df=pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df
  
  st.write('**X**')
  X= df.drop('species', axis=1)
  X

  st.write('**y**')
  y= df.species
  y
  
with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

#Data preprations
with st.sidebar:
  st.header('Input Features')
  #"island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  island= st.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
  sex= st.selectbox('Sex', ('Male', 'Female'))
  bill_length_mm= st.slider('Bill length (mm)', 39.1 , 59.6,43.9)
  bill_depth_mm= st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  body_mass_g= st.slider('Body mass(g)' , 2700.0, 6300.0, 4207.0)
  flipper_length_mm=st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  #Create dataframe for input features
  
data = {'island': island,
         'sex': sex,
         'bill_length_mm': bill_length_mm,
         'bill_depth_mm': bill_depth_mm,
         'body_mass_g': body_mass_g, 
         'flipper_length_mm': flipper_length_mm}
         
input_df= pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, X], axis=0)

with st.expander('Input features'):
  st.write('**Input Features**')
  input_df
  st.write('**Combined Data**')
  input_penguins
  
encode= ['island','sex']
df_penguins= pd.get_dummiess(input_penguins, prefix= encode)
df_penguins
  
