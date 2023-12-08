import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained logistic regression model
lr_model = joblib.load('trained_lr_model.joblib')  # Update the path if needed

# Function to clean data as per the provided script
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Function to preprocess the input data
def preprocess_data(input_data):
    ss = pd.DataFrame({
        "income": np.where(input_data["income"] > 9, np.nan, input_data["income"]),
        "education": np.where(input_data["education"] > 8, np.nan, input_data["education"]),
        "parent": clean_sm(input_data["parent"]),
        "married": np.where(input_data["married"] == 1, 1, 0),
        "female": np.where(input_data["female"] == 2, 1, np.where(input_data["female"] == 1, 0, np.nan)),
        "age": np.where(input_data["age"] > 98, np.nan, input_data["age"])
    })
    return ss.dropna()

# Streamlit app
st.title('LinkedIn Usage Prediction')

# User inputs
income_options = [
    'Less than $10,000', '10 to under $20,000', '20 to under $30,000', '30 to under $40,000',
    '40 to under $50,000', '50 to under $75,000', '75 to under $100,000', '100 to under $150,000',
    '$150,000 or more', 'Don\'t know', 'Refused'
]
income = st.selectbox('Income', options=income_options, index=0)
income = income_options.index(income) + 1  # Convert to numeric value

education_options = [
    'Less than high school', 'High school incomplete', 'High school graduate', 'Some college, no degree',
    'Two-year associate degree', 'Four-year college/university degree', 'Some postgraduate or professional schooling',
    'Postgraduate or professional degree', 'Don\'t know', 'Refused'
]
education = st.selectbox('Education', options=education_options, index=0)
education = education_options.index(education) + 1  # Convert to numeric value

# Updated parent status options
parent_options = ['Yes', 'No', 'Don\'t know', 'Refused']
parent = st.selectbox('Are you a parent of a child under 18 living in your home?', options=parent_options, index=0)
parent = parent_options.index(parent) + 1

married_status = ['Married', 'Living with a partner', 'Divorced', 'Separated', 'Widowed', 'Never been married', 'Don\'t know', 'Refused']
married = st.selectbox('Marital Status', options=married_status, index=0)
married = 1 if married_status.index(married) == 0 else 0

gender_options = ['Male', 'Female', 'Other', 'Don\'t know', 'Refused']
gender = st.selectbox('Gender', options=gender_options, index=0)
gender = gender_options.index(gender) + 1
  
age = st.slider('Age', 18, 97, 18)

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
        prediction_probability = lr_model.predict_proba(processed_data)[0][1]  # Probability of being a LinkedIn user
        prediction_text = 'Uses LinkedIn' if prediction[0] == 1 else 'Does not use LinkedIn'
        st.write(f'Prediction: {prediction_text}')
        st.write(f'Probability of using LinkedIn: {prediction_probability:.2f}%')
