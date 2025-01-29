#Importing the necessary libraries

import joblib
import numpy as np
import streamlit as st

@st.cache_resource()

def load_model():
    return joblib.load('Random_Forest_Model.pkl')


st.cache_resource.clear()

st.title('Gender Classification Model')
st.subheader('This model will help predict the gender of indiividuals based on the given features')

#Loading the model
model = load_model()

#User inputs

if model:
    st.header('Please enter the following details:')

nose_wide = st.selectbox('Has a wide nose?', options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
nose_wide_value = nose_wide[0]

distance_nose_to_lip_long = st.selectbox('Has Long Distance between nose and lips?', options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
distance_nose_to_lip_long_value = distance_nose_to_lip_long[0]

nose_long = st.selectbox('Has a long nose?', options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
nose_long_value = nose_long[0]

lips_thin = st.selectbox('Has thin lips?', options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
lips_thin_value = lips_thin[0]

forehead_width_cm = st.number_input('What is your forehead width?', min_value=  0.00, max_value = 15.50, value = 0.00)
forehead_height_cm = st.number_input('What is your forehead height?', min_value = 0.00, max_value = 7.10, value = 0.00)

user_input = np.array([nose_wide_value, distance_nose_to_lip_long_value, nose_long_value, lips_thin_value, forehead_width_cm, forehead_height_cm])

#Making prediction

if st.button('Predict Gender', key = 'Predict Button'):
    prediction = model.predict(user_input.reshape(1,-1))
    gender = 'Male' if prediction[0] == 0 else 'Female'
    st.subheader(f'The predicted gender is {gender}')

