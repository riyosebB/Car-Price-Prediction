import streamlit as st
import pickle
import numpy as np
import os


folder_path = os.path.dirname(os.path.abspath(__file__))

# Load the trained models
with open('Car_Sale_Price_Prediction_Linear_Regression.pkl', 'rb') as file:
    linear_model = pickle.load(file)

with open('Car_Sale_Price_Prediction_Elastic_Net.pkl', 'rb') as file:
    elastic_net_model = pickle.load(file)

with open('Car_Sale_Price_Prediction_Lasso_Regression.pkl', 'rb') as file:
    lasso_model = pickle.load(file)

with open('Car_Sale_Price_Prediction_Ridge_Regression.pkl', 'rb') as file:
    ridge_model = pickle.load(file)

# To map the input value make to this, thus make the make_value 
make = {'Acura': 14132.190406711654,
 'Audi': 15431.180469715699,
 'BMW': 16439.04020624885,
 'Buick': 10636.497933884297,
 'Cadillac': 13100.688439220354,
 'Chevrolet': 11237.762227550524,
 'Chrysler': 10863.55921133703,
 'Daewoo': 400.0,
 'Dodge': 11059.63869762292,
 'FIAT': 9946.72229299363,
 'Ford': 14110.88865781475,
 'GMC': 15930.100010650762,
 'Geo': 588.4615384615385,
 'HUMMER': 15143.863402061856,
 'Honda': 11439.088529457727,
 'Hyundai': 10948.934658420283,
 'Infiniti': 19620.958768592183,
 'Isuzu': 1786.6935483870968,
 'Jaguar': 11847.536832412523,
 'Jeep': 14222.729564176583,
 'Kia': 11781.281188453098,
 'Land Rover': 19108.979357798165,
 'Lexus': 18792.61960440417,
 'Lincoln': 16264.242649795311,
 'MINI': 12315.771175726928,
 'Maserati': 21921.69811320755,
 'Mazda': 10349.674671948502,
 'Mercedes-Benz': 17693.09028651293,
 'Mercury': 4058.364116094987,
 'Mitsubishi': 8385.625125125125,
 'Nissan': 11791.637624325092,
 'Oldsmobile': 1004.7619047619048,
 'Plymouth': 10598.611111111111,
 'Pontiac': 3884.653644592663,
 'Porsche': 15413.194444444445,
 'Ram': 22200.50522839846,
 'Saab': 3662.130801687764,
 'Saturn': 3482.20495745468,
 'Scion': 9686.45141822571,
 'Subaru': 15547.199593082401,
 'Suzuki': 3932.3036896877957,
 'Toyota': 12430.42351089653,
 'Volkswagen': 9405.541581819663,
 'Volvo': 11761.163938897982,
 'airstream': 71000.0,
 'audi': 16066.666666666666,
 'bmw': 18425.0,
 'buick': 6452.083333333333,
 'cadillac': 5069.0,
 'chevrolet': 4892.96875,
 'chrysler': 2898.6979166666665,
 'dodge': 2522.3684210526317,
 'ford': 5844.513888888889,
 'gmc': 6903.846153846154,
 'honda': 4884.009009009009,
 'hyundai': 1555.5555555555557,
 'jeep': 6627.4647887323945,
 'kia': 5875.0,
 'land rover': 10772.782258064517,
 'lexus': 13218.981481481482,
 'lincoln': 21135.185185185186,
 'maserati': 32200.0,
 'mazda': 4230.285714285715,
 'mercury': 2421.0,
 'mitsubishi': 7707.589285714285,
 'nissan': 5058.064516129032,
 'oldsmobile': 765.0,
 'plymouth': 625.0,
 'pontiac': 3923.3333333333335,
 'porsche': 20170.0,
 'smart': 6288.917525773196,
 'subaru': 3830.2631578947367,
 'suzuki': 5562.5,
 'toyota': 9214.191176470587,
 'volkswagen': 5481.25}



# Define a function to make predictions
def predict_price(model, year, mmr, odometer, is_automatic, make_value):
    # Prepare the input data
    input_data = np.array([[year, mmr, odometer, is_automatic, make_value]])
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
model = {
    'Elastic Net': elastic_net_model,
    'Linear Regression': linear_model,
    'Lasso': lasso_model,
    'Ridge': ridge_model
}[model_option]

# Input fields
year = st.number_input('Year', min_value=1900, max_value=2024, value=2024)
mmr = st.number_input('MMR', min_value=0, value=12000)
odometer = st.number_input('Odometer', min_value=0, value=15000)
is_automatic = st.selectbox('Is Automatic', options=[0, 1])
make_input = st.text_input('Make').strip().lower()

# Convert the input make to the corresponding value
make_value = make.get(make_input)

if make_value is None:
    pass
else:
    # Button to predict
    if st.button('Predict Price'):
        predicted_price = predict_price(model, year, mmr, odometer, is_automatic, make_value)
        st.write(f'The predicted selling price is ${predicted_price:,.2f}')

# Key visualizations
st.title('Model Performance Metrics')

# Metrics and visualization dictionary
metrics = {
    'Linear Regression': {"Root Mean Squared Error": 1715.1374645734595, "R^2 Score": 0.9481667923252897, "Graph": 'Linear.png'},
    'Ridge': {"Root Mean Squared Error": 1715.1366107909842, "R^2 Score": 0.9481668439296354, "Graph": 'Ridge.png'},
    'Lasso': {"Root Mean Squared Error": 1715.1373206333474, "R^2 Score": 0.9481668010253239, "Graph": 'Lasso.png'},
    'Elastic Net': {"Root Mean Squared Error": 1715.234130396293, "R^2 Score": 0.9481609494806543, "Graph": 'ElasticNet.png'},
}

# Display selected model metrics
if model_option in metrics:
    metrics = metrics[model_option]
    st.header(model_option)
    st.write(f'Root Mean Squared Error: {metrics["Root Mean Squared Error"]}')
    st.write(f'R^2 Score: {metrics["R^2 Score"]}')
    st.image(metrics["Graph"])
else:
    st.write("Please select a model.")
