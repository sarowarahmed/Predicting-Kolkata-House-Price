import streamlit as st
import pandas as pd

st.title('🏡Predicting Kolkata House Price')

st.info('This app builds a machine learning model to predict the house price in kolkata')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/sarowarahmed/Predicting-Kolkata-House-Price/master/Kolkata%20House%20Price.csv')
  df

  # Separate features and target
  st.write('**X**')
  X = df.drop(['Property_ID', 'PRICE(₹)'], axis=1)
  X
  st.write('**y**')
  y = df['PRICE(₹)']
  y

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='Location', y='Flood_Zone', color='PRICE(₹)')
 
#Data Preparation
with st.sidebar:
  st.header('Input Features')
  # Location,Property_Type,Number_of_Bedrooms,Age_of_Property,Furnishing_Status,Square_Footage,Crime_Rate_in_Area,Air_Quality_Index,Ownership_Type
  Location = st.selectbox('Location', ('Salt Lake', 'New Town', 'Ballygunge', 'Alipore', 'Howrah', 'Madhyamgram'))
  Property_Type = st.selectbox('Property Type', ('Apartment', 'Villa', 'Independent House'))
  Number_of_Bedrooms = st.slider('BHK', 1,4)
  Age_of_Property = st.slider('Property Age(Y)', 0,75,38)
  Square_Footage = st.number_input('Insert Sq.ft',500,5000,2250)
  Furnishing_Status = st.selectbox('Furnishing Status', ('Furnished', 'Semi-furnished', 'Unfurnished'))

  #create DataFrame for Input Feature
  data = {'Location':Location,
          'Property_Type': Property_Type,
          'Number_of_Bedrooms': Number_of_Bedrooms,
          'Age_of_Property': Age_of_Property,
          'Square_Footage': Square_Footage,
          'Furnishing_Status': Furnishing_Status}
  input_df = pd.DataFrame(data, index=[0])
  input_concatenate = pd.concat([input_df, X], axis=0)

with st.expander('Input Features'):
  st.write('**Input Housing Data**')
  input_df
  st.write('**Combined Housing Data**')
  input_concatenate
