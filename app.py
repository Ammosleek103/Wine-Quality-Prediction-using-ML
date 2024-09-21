import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('extra_trees_model.pkl')

# Define a function to make predictions
def predict_wine_quality(features):
    prediction = model.predict([features])
    return prediction[0]

# Streamlit app layout
st.title('Wine Quality Prediction App')

# Define feature input fields
st.header('Enter Wine Features:')
fixed_acidity = st.number_input('Fixed Acidity', value=7.4)
volatile_acidity = st.number_input('Volatile Acidity', value=0.7)
citric_acid = st.number_input('Citric Acid', value=0.0)
residual_sugar = st.number_input('Residual Sugar', value=1.9)
chlorides = st.number_input('Chlorides', value=0.076)
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', value=11.0)
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', value=34.0)
density = st.number_input('Density', value=0.9978)
pH = st.number_input('pH', value=3.51)
sulphates = st.number_input('Sulphates', value=0.56)
alcohol = st.number_input('Alcohol', value=9.4)

# Organize the inputs into a list for the model
features = [
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, 
    density, pH, sulphates, alcohol
]

# When the user clicks the 'Predict' button, make the prediction
if st.button('Predict Wine Quality'):
    result = predict_wine_quality(features)
    st.success(f'The predicted wine quality is: {result}')
