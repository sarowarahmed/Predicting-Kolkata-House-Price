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
 
