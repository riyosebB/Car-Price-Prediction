# Car-Price-Prediction


## Overview

This repository contains a machine learning project for predicting car prices using a $ different models: 
   
     1 Linear Regression model
     2 Ridge Regression model
     3 Lasso Regression model
     4 ElasticNet Regression model
     
The goal is to predict the selling price of a car based on these features. The models are trained & then saved as pickle file in the format Car_Sale_Price_Prediction_Model_Name.pkl

## Dataset

The model is trained on a dataset containing various features related to cars, such as year, odometer reading, make, and more. 


## Model 
The 4 models uses the below features based on the correaltion with the target feature.

    Year: The year the car was manufactured.
    MMR: Manheim Market Report - a value used in the automotive industry.
    Odometer: The number of miles the car has been driven.
    Is Automatic: A binary feature indicating if the car has an automatic transmission.
    Make: The brand or manufacturer of the car.

The target variable is the Selling Price.

The model is built using the following pipeline:

    Data Preprocessing: Handling missing values, encoding categorical variables, and scaling features.
    Model Training: Using Scikit-learn's LinearRegression, RidgeRegression, LassoRegression, ElasticNetRegression class.
    Evaluation: Metrics such as RMSE and R^2 Score are used to evaluate the model's performance.

## Streamlit Application

Link to the app:

    https://car-price-prediction-rocbsgtkbazrgxvnglwj4d.streamlit.app/

