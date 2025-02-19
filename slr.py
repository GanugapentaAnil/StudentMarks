# -*- coding: utf-8 -*-
"""
Simple Linear Regression Model Deployment with Streamlit
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load the trained model
loaded_model = pickle.load(open('linear.sav', 'rb'))

# Prediction function
def predict_scores(hours):
    input_data = np.array([[hours]])  # Convert input to a 2D array
    prediction = loaded_model.predict(input_data)
    return prediction[0]  # Return the predicted score

# Streamlit UI
def main():
    st.title("Student Score Predictor")
    st.write("Enter the number of study hours to predict the score.")

    # Input from the user
    hours = st.number_input("Insert number of study hours", min_value=0.0, step=0.1)

    # Prediction button
    if st.button("Predict Score"):
        predicted_score = predict_scores(hours)
        st.success(f"Predicted Score: {predicted_score:.2f}")

if __name__ == "__main__":
    main()
