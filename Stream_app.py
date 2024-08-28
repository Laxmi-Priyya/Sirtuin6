import numpy as np
import pandas as pd
import streamlit as st 
import joblib
import tensorflow as tf
model=tf.keras.models.load_model('sirtuin_model.h5')
scaler=joblib.load('scaler.pkl')

st.title("Identification of High / Low BFE in Sirtuin")
st.sidebar.header("Enter input features")
def user_input_features():
    feature_1 = st.sidebar.number_input('SC-5', value=0.0)
    feature_2 = st.sidebar.number_input('SP-6', value=0.0)
    feature_3 = st.sidebar.number_input('SHBd', value=0.0)
    feature_4 = st.sidebar.number_input('minHaaCH', value=0.0)
    feature_5 = st.sidebar.number_input('maxwHBa', value=0.0)
    feature_6 = st.sidebar.number_input('FMF', value=0.0)

    data = {
        'SC-5': feature_1,
        'SP-6': feature_2,
        'SHBd': feature_3,
        'minHaaCH': feature_4,
        'maxwHBa': feature_5,
        'FMF': feature_6,
    }

    features = pd.DataFrame(data, index=[0])
    return features
user=user_input_features()
if st.button("Predict"):
    # Standardize user input
    user_scaled = scaler.transform(user)
    
    # Predict the class (0 for Kecimen, 1 for Besni)
    predicted_class = model.predict(user_scaled)[0][0]
    
    # Display the prediction
    if predicted_class < 0.5:
        st.markdown("<h2 style='color: #28a745;'>Prediction: High BFE</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: #d9534f;'>Prediction : Low BFE </h2>", unsafe_allow_html=True)



# Display the user inputs
st.subheader("User Input Features")
st.write(user)
