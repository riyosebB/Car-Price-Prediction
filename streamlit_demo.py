import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import os


# Load data for metrics and visualizations
folder_path = os.path.dirname(os.path.abspath(__file__))
file_path = folder_path + "/car_sale_prediction.csv"
df = pd.read_csv(file_path)

# Load the trained models
with open('Car Sale Price Prediction Linear Regression.pkl', 'rb') as file:
    linear_model = pickle.load(file)

with open('Car Sale Price Prediction Elastic Net.pkl', 'rb') as file:
    elastic_net_model = pickle.load(file)

with open('Car Sale Price Prediction Lasso Regression.pkl', 'rb') as file:
    lasso_model = pickle.load(file)



with open('Car Sale Price Prediction Ridge Regression.pkl', 'rb') as file:
    ridge_model = pickle.load(file)




# Define a function to make predictions
def predict_price(model, year, mmr, odometer, is_automatic):
    # Prepare the input data
    input_data = np.array([[year, mmr, odometer, is_automatic]])
    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app layout
st.title('Car Price Prediction')

# Model selection
model_option = st.selectbox(
    'Select Model',
    ['Linear Regression', 'Elastic Net', 'Lasso', 'Ridge']
)

# Set the model based on user selection
if model_option == 'Elastic Net':
    model = elastic_net_model
elif model_option == 'Linear Regression':
    model = linear_model
elif model_option == 'Lasso':
    model = lasso_model
else:
    model = ridge_model

# Input fields
year = st.number_input('Year', min_value=1900, max_value=2024, value=2024)
mmr = st.number_input('MMR', min_value=0, value=12000)
odometer = st.number_input('Odometer', min_value=0, value=15000)
is_automatic = st.selectbox('Is Automatic', options=[0, 1])

# Button to predict
if st.button('Predict Price'):
    predicted_price = predict_price(model, year, mmr, odometer, is_automatic)
    st.write(f'The predicted selling price is ${predicted_price:,.2f}')

# Key visualizations
st.title('Model Performance Metrics')


lr = {"Model": "Linear Regression","Root Mean Squared Error": 1715.1374645734595 , "R^2 Score": 0.9481667923252897, "Graph": 'Linear.png'}

ridge_re = {"Model": "Ridge Regression","Root Mean Squared Error": 1715.1366107909842, "R^2 Score": 0.9481668439296354, "Graph": 'Ridge.png'}

lasso_re = {"Model": "Lasso Regression","Root Mean Squared Error": 1715.1373206333474, "R^2 Score": 0.9481668010253239, "Graph": 'Lasso.png'}

en = {"Model": "Elastic Net Regression","Root Mean Squared Error": 1715.234130396293, "R^2 Score": 0.9481609494806543, "Graph": 'ElasticNet.png'}

# Set the model based on user selection
# Set the model based on user selection
if model_option == 'Elastic Net':
    st.header(en["Model"])
    st.write(f'Root Mean Squared Error: {en["Root Mean Squared Error"]}')
    st.write(f'R^2 Score: {en["R^2 Score"]}')
    st.image(en["Graph"])
elif model_option == 'Linear Regression':
    st.header(lr["Model"])
    st.write(f'Root Mean Squared Error: {lr["Root Mean Squared Error"]}')
    st.write(f'R^2 Score: {lr["R^2 Score"]}')
    st.image(lr["Graph"])
elif model_option == 'Lasso':
    st.header(lasso_re["Model"])
    st.write(f'Root Mean Squared Error: {lasso_re["Root Mean Squared Error"]}')
    st.write(f'R^2 Score: {lasso_re["R^2 Score"]}')
    st.image(lasso_re["Graph"])
elif model_option == 'Ridge':
    st.header(ridge_re["Model"])
    st.write(f'Root Mean Squared Error: {ridge_re["Root Mean Squared Error"]}')
    st.write(f'R^2 Score: {ridge_re["R^2 Score"]}')
    st.image(ridge_re["Graph"])
else:
    st.write("Please select a model.")







