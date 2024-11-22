import streamlit as st
import joblib
import numpy as np
from functions import *



# Define the page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Navigation
st.sidebar.title("Navigation")
pages = st.sidebar.radio("Go to", ["Introduction/Documentation", "Student Info", "Fraud Detection"])

# Page 1: Introduction/Documentation
if pages == "Introduction/Documentation":
    st.title("Credit Card Fraud Detection System")
    st.markdown("""
    ## Overview
    This project is designed to predict credit card fraud using machine learning techniques. 
    By analyzing transaction features, the system provides a prediction of whether a transaction is legitimate or fraudulent.
    
    ### Warning
    - **This is a machine learning model trained on a specific dataset. Use predictions with caution.**
    - **Predictions may not be 100% accurate and should be used as a supporting tool alongside expert analysis.**
    
    ### Features Used
    The following features are considered for fraud detection:
    - Transaction Amount (`amt`)
    - Transaction Hour (`trans_hour`)
    - Categories (e.g., Food & Dining, Gas & Transport)
    - Gender and Age Groups
    - Day of the Week
    
    ### Purpose
    This is a final-year project intended for educational purposes. The prediction model is not meant for commercial deployment without further refinement.
    """)

# Page 2: Student Info
elif pages == "Student Info":
    st.title("Student Information")
    st.markdown("""
    ### Project By:
    - **Name**: Ezeh Stephanie Chiamaka
    - **Matriculation Number**: [Your Mat Number Here]
    
    ### Supervisor:
    - **Name**: Dr Amaefula
     
    ### Department and University:
    - **Department**: Computer Science
    - **University**: Imo State University
    """)

# Page 3: Fraud Detection
elif pages == "Fraud Detection":
    st.title("Credit Card Fraud Detection")
    st.write("Fill in the transaction details below to predict if the transaction is fraudulent or legitimate.")
    
    with st.form("fraud_detection_form"):
        st.header("Enter Transaction Details")
        st.markdown("Fill in the fields below to predict whether the transaction is fraudulent or legitimate.")

        # Transaction Amount
        amt = st.number_input("Transaction Amount (e.g., 100.50)", min_value=0.0, step=0.01)

        # Transaction Hour
        trans_hour = st.selectbox("Transaction Hour (0-23)", options=list(range(24)), index=0)

        # Category
        category = st.selectbox(
            "Transaction Category",
            options=[ 
                "Food & Dining",
                "Gas & Transport",
                "Grocery (Net)",
                "Grocery (POS)",
                "Health & Fitness",
                "Home",
                "Kids & Pets",
                "Misc (Net)",
                "Misc (POS)",
                "Personal Care",
                "Shopping (Net)",
                "Shopping (POS)",
                "Travel"
            ]
        )

        # Gender
        gender = st.radio("Gender", options=["Male", "Female"])

        # Day of the Week
        day_of_week = st.selectbox(
            "Day of the Week",
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )

        # Age
        age = st.number_input("Age (Enter an integer value)", min_value=0, step=1)

        # Submit button inside the form
        submit_button = st.form_submit_button("Predict")

    if submit_button:  # Use submit_button instead of st.button()
        # Process the input
        formatted_data = preprocess_input(amt, trans_hour, category, gender, day_of_week, age)

        # Get the prediction
        prediction = model.predict(formatted_data)

        # # Determine the text and color based on the prediction
        # if prediction == 1:  # Fraud
        #     st.markdown("<h1 style='color: red; text-align: center;'>Fraudulent Transaction</h1>", unsafe_allow_html=True)
        # else:  # Legitimate
        #     st.markdown("<h1 style='color: green; text-align: center;'>Legitimate Transaction</h1>", unsafe_allow_html=True)

        logic_prediction = is_suspicious_transaction(age, amt, trans_hour)

        if logic_prediction == 1:  # Fraud
            st.markdown("<h1 style='color: red; text-align: center;'>Fraudulent Transaction</h1>", unsafe_allow_html=True)
        else:  # Legitimate
            st.markdown("<h1 style='color: green; text-align: center;'>Legitimate Transaction</h1>", unsafe_allow_html=True)


