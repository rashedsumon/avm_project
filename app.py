import streamlit as st
from models.avm_model import train_model, predict_price
import pandas as pd

def main():
    st.title("Automated Valuation Model (AVM) for Real Estate")
    
    st.sidebar.header("Property Details")
    
    # User input for property features
    location = st.sidebar.text_input("Location")
    area = st.sidebar.number_input("Area (sq ft)", min_value=0)
    rooms = st.sidebar.number_input("Number of rooms", min_value=0)
    year_built = st.sidebar.number_input("Year Built", min_value=1900, max_value=2023)
    renovation_level = st.sidebar.selectbox("Renovation Level", ["None", "Minor", "Major"])
    
    # Load the trained model
    model = train_model()  # This will load the pre-trained model
    
    if st.sidebar.button("Predict Price"):
        # Predict the price using the input data
        prediction = predict_price(model, location, area, rooms, year_built, renovation_level)
        st.write(f"Predicted Price: ${prediction:,.2f}")

if __name__ == "__main__":
    main()
