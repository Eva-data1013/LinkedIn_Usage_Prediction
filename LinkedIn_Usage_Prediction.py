import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Function to preprocess the input data
lr_model = joblib.load('trained_lr_model.joblib')  

# Function to clean data
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Function to preprocess the input data
def preprocess_data(input_data):
    input_data['income'] = np.where(input_data['income'].isin([10, 11]), -1, input_data['income'])
    input_data['education'] = np.where(input_data['education'] == 10, -1, input_data['education'])
    input_data['female'] = np.where(input_data['female'] == 3, -1, 
                                    np.where(input_data['female'] == 2, 1, 
                                             np.where(input_data['female'] == 1, 0, -1)))
    input_data['parent'] = clean_sm(input_data['parent'])
    input_data['married'] = np.where(input_data['married'] == 1, 1, 0)
    input_data['age'] = np.where(input_data['age'] > 98, np.nan, input_data['age'])
    return input_data

# Streamlit app
st.title('LinkedIn Usage Prediction')

# User inputs
income_options = [
    'Less than $10,000', '10 to under $20,000', '20 to under $30,000', '30 to under $40,000',
    '40 to under $50,000', '50 to under $75,000', '75 to under $100,000', '100 to under $150,000',
    '$150,000 or more', 'Don\'t know', 'Refused'
]
income = st.selectbox('Household Income:', options=income_options, index=0)
income = income_options.index(income) + 1  

education_options = [
    'Less than high school', 'High school incomplete', 'High school graduate', 'Some college, no degree',
    'Two-year associate degree', 'Four-year college/university degree', 'Some postgraduate or professional schooling',
    'Postgraduate or professional degree', 'Don\'t know', 'Refused'
]
education = st.selectbox('Highest Education Level:', options=education_options, index=0)
education = education_options.index(education) + 1  

parent_options = ['Yes', 'No', 'Don\'t know', 'Refused']
parent = st.selectbox('Do You Have Children Under 18 at Home?', options=parent_options, index=0)
parent = parent_options.index(parent) + 1

married_status = ['Married', 'Living with a partner', 'Divorced', 'Separated', 'Widowed', 'Never been married', 'Don\'t know', 'Refused']
married = st.selectbox('Marital Status:', options=married_status, index=0)
married = 1 if married_status.index(married) == 0 else 0

gender_options = ['Male', 'Female', 'Other', 'Don\'t know', 'Refused']
gender = st.selectbox('Gender:', options=gender_options, index=0)
gender = gender_options.index(gender) + 1

age = st.number_input('Your Age:', min_value=18, max_value=97, value=18, step=1)

# Predict button
if st.button('Predict LinkedIn Usage'):
    input_data = pd.DataFrame([[income, education, parent, married, gender, age]],
                              columns=['income', 'education', 'parent', 'married', 'female', 'age'])
    processed_data = preprocess_data(input_data)
    
    # Check if data is not dropped due to preprocessing
    if processed_data.empty:
        st.write("Invalid input data.")
    else:
        prediction = lr_model.predict(processed_data)
        prediction_probability = lr_model.predict_proba(processed_data)[0][1]  
        prediction_text = 'Uses LinkedIn' if prediction[0] == 1 else 'Does not use LinkedIn'
        st.write(f'Prediction: {prediction_text}')
        st.write(f'Probability of using LinkedIn: {prediction_probability * 100:.2f}%')
