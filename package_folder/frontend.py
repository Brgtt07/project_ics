import streamlit as st
import requests

# Cloud Run API URL
API_URL = 'https://project-ics-210911899890.europe-west1.run.app/predict'

st.title("🌍 Find Your Ideal Country to Live!")

# User inputs (at the moment just to test)
climate = st.selectbox("Preferred Climate:", ["Hot", "Cold", "Moderate"])
cost_of_living = st.slider("Cost of Living (1 - Low, 5 - High)", 1, 5, 3)
safety = st.slider("Safety Level (1 - Low, 5 - High)", 1, 5, 3)
internet_quality = st.slider("Internet Quality (1 - Poor, 5 - Excellent)", 1, 5, 3)

# Submit button to trigger API call
if st.button("Find My Ideal Country"):
    # Prepare data for API call
    data = {
        "climate": climate,
        "cost_of_living": cost_of_living,
        "safety": safety,
        "internet_quality": internet_quality
    }
    
    # Make API call
    try:
        response = requests.post(API_URL, json=data)
        response_data = response.json()
        
        # Display response
        st.success("Here are your results!")
        st.json(response_data)
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        st.info("Note: Make sure the API server is running on https://project-ics-210911899890.europe-west1.run.app")

# Test API button
if st.button("Test API"):
    test_endpoint = "https://project-ics-210911899890.europe-west1.run.app/predict"
    # Prepare data for API call
    greeting = "hello"  # or any other value you want to test
    # Make API call
    try:
        response = requests.get(f"{test_endpoint}?greeting={greeting}")
        response_data = response.json()
        
        # Display response
        st.success("Here are your results!")
        st.json(response_data)
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        st.info("Note: Make sure the API server is running on https://project-ics-210911899890.europe-west1.run.app")
