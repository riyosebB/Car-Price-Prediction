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
    ['Elastic Net', 'Lasso', 'Ridge']
)

# Set the model based on user selection
if model_option == 'Elastic Net':
    model = elastic_net_model
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
st.header('Model Performance Metrics')

# Ensure to use the same features as your model
X_features = df[['year', 'mmr', 'odometer', 'Is Automatic']]
y_true = df['sellingprice']

# Predictions for the entire dataset using the selected model
y_pred = model.predict(X_features)

# Compute metrics
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
st.write(f'Root Mean Squared Error: {rmse:.2f}')
st.write(f'R-squared (R²): {r2:.2f}')

# Data Distribution Plot
st.header('Data Distribution')
fig, ax = plt.subplots()
sns.histplot(df['sellingprice'], kde=True, ax=ax)
ax.set_title('Distribution of Selling Prices')
st.pyplot(fig)

# Prediction Results
st.header('Prediction Results')
st.write(df.head())
