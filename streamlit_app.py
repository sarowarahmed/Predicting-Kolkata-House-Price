import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

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
  Furnishing_Status = st.selectbox('Furnishing Status', ('Furnished', 'Semi-furnished', 'Unfurnished'))
  Square_Footage = st.number_input('Insert Sq.ft',500,5000,2250)
  Crime_Rate_in_Area = st.slider('Crime Rate in Area', 0,10)
  Air_Quality_Index = st.slider('AQI', 50,300,150)
  Ownership_Type = st.selectbox('Ownership Type', ('Freehold', 'Leasehold'))
  Flood_Zone = st.selectbox('Flood Zone', ('Zone A', 'Zone B','Zone C','Zone D'))

  #create DataFrame for Input Feature
  data = {'Location':Location,
          'Property_Type': Property_Type,
          'Number_of_Bedrooms': Number_of_Bedrooms,
          'Age_of_Property': Age_of_Property,
          'Furnishing_Status': Furnishing_Status,
          'Square_Footage': Square_Footage,
          'Crime_Rate_in_Area': Crime_Rate_in_Area,
          'Air_Quality_Index': Air_Quality_Index,
          'Ownership_Type': Ownership_Type,
          'Flood_Zone': Flood_Zone}
  input_df = pd.DataFrame(data, index=[0])
  input_concatenate = pd.concat([input_df, X], axis=0)

with st.expander('Input Features'):
  st.write('**Input Housing Data**')
  input_df
  st.write('**Combined Housing Data**')
  input_concatenate

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Number_of_Bedrooms', 'Age_of_Property', 'Square_Footage', 'Crime_Rate_in_Area', 'Air_Quality_Index']),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), ['Location', 'Property_Type', 'Furnishing_Status', 'Ownership_Type', 'Flood_Zone'])
    ])

# Create a pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the model
model.fit(X, y)

# Make predictions
if st.button('Predict House Price'):
    # Preprocess the input
    input_processed = model.named_steps['preprocessor'].transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_processed)
    
    # Display Predicted Housing Price
    st.subheader('Predicted House Price')
    st.success(f'The predicted price for the house is ₹{prediction[0]:,.2f}')
