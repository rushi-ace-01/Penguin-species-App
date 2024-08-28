import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This app predicts outcomes based on machine learning model!')

with st.expander('Data'):
  st.write('**Raw Data**')
  df=pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df
  
  st.write('**X**')
  X_raw= df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw
  
with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

#Data preprations
with st.sidebar:
  st.header('Input Features')
  #"island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  island= st.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
  sex= st.selectbox('Sex', ('male', 'female'))
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
input_penguins = pd.concat([input_df, X_raw], axis=0)

#Encode X
encode= ['island','sex']
df_penguins= pd.get_dummies(input_penguins, prefix= encode)
input_row= df_penguins[:1]

#Encode y
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo':2}
def target_encode(val):
  return target_mapper[val]
y= y_raw.apply(target_encode)
y
          
with st.expander('Input features'):
  st.write('**Input Penguin**')
  input_df
  st.write('**Combined Penguin Data**')
  input_penguins
  st.write('**Encoded input penguin**')
  input_row
  
