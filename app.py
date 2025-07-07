#!/usr/bin/env python3.11

import streamlit as st 
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd 
import numpy as np
import dill



# Load transformer
with open('model.pkl', 'rb') as f:
    transformer = dill.load(f)

# Load trained Keras model
model = load_model('best_model.h5')

# Streamlit UI
st.title('📊 Customer Churn Prediction')

df = pd.read_csv('Churn_Modelling.csv')




# CreditScore	Geography	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary

credit_score = st.number_input(label='Credit Score')
Geography = st.selectbox('Geography',options=df['Geography'].unique())
Gender = st.selectbox('Gender',options=df['Gender'].unique())
Age = st.slider('Age',min_value=17,max_value=98)
Tenure = st.slider('Tenure',min_value=0,max_value=10)
Balance = st.number_input('Balance')
NumOfProduct = st.slider("Number Of Product",min_value=1,max_value=10)
HasCrCard = st.selectbox('Has Credit Card' , ['NO' ,'YES'])
IsActiveMember = st.selectbox('Is Active Member' , ['NO' ,'YES'])
EstimatedSalary = st.number_input('Estimated Salary')


button = st.button('Click here')

if button:
    input_dict = {
        'CreditScore': [credit_score],
        'Geography': [Geography],
        'Gender': [Gender],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProduct],
        'HasCrCard': [1 if HasCrCard == 'YES' else 0],
        'IsActiveMember': [1 if IsActiveMember == 'YES' else 0],
        'EstimatedSalary': [EstimatedSalary]
    }

    input_df = pd.DataFrame(input_dict)

    # Transform data
    data = transformer.transform(input_df)

    # Predict
    prediction = model.predict(data)


    

    if prediction[0][0] < 0.5:
        st.write('Customer Is Not Likely To Churn ❌')
        st.error(f'Churn Probability : {prediction[0][0]:.2f}')
    else:
        st.write('Customer Is Likely To Churn ✅')
        st.success(f'Churn Probability: {prediction[0][0]:.2f}')



