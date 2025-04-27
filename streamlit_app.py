import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
model = pickle.load(open('best_car_price_model.pkl', 'rb'))
le_maker = pickle.load(open('le_maker.pkl', 'rb'))
le_model = pickle.load(open('le_model.pkl', 'rb'))

st.title("Car Price Predictor")

# Input fields
car_maker_input = st.text_input("Car Maker (e.g., Toyota)")
model_input = st.text_input("Car Model (e.g., Camry)")
year = st.number_input("Year", min_value=1990, max_value=2025, step=1)
kilometers = st.number_input("Kilometers Driven", min_value=0)
condition = st.selectbox("Condition (Used = 1,  New = 0)", ["1", "0"])  # We'll map later
transmission = st.selectbox("Transmission (Automatic = 1 , Manual = 0 )", ["1", "0"])  # We'll map later

# Mapping text to numbers
if st.button("Predict"):
    try:
        car_maker_encoded = le_maker.transform([car_maker_input.lower()])[0]
        model_encoded = le_model.transform([model_input.lower()])[0]
    except ValueError:
        st.error("Invalid car maker or model. Please enter a valid one.")
    else:
        
        # Create input DataFrame
        input_df = pd.DataFrame({
            'car_maker': [car_maker_encoded],
            'model': [model_encoded],
            'year': [year],
            'condition': [condition],
            'kilometers': [kilometers],
            'transmission': [transmission]
        })
        
        prediction = model.predict(input_df)
        st.success(f"Estimated Car Price: {prediction[0]:,.2f} SAR")
