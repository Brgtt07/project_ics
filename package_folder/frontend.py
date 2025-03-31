import streamlit as st
import requests

# Cloud Run API URL
API_URL = 'https://project-ics-210911899890.europe-west1.run.app/predict'
# Local API URL for testing
api_local = 'http://localhost:8000/predict'

st.title("🌍 Find Your Ideal Country to Live!")
st.write("")

# User selects the importance of each feature (1-10 scale)
st.subheader("1️⃣ Rate the Importance of Each Category (1 = Not Important, 10 = Very Important)")
weights = {
    "Safety": st.slider("Safety", 1, 10, 5),
    "Cost of Living": st.slider("Cost of Living", 1, 10, 5),
    "Health Care Index": st.slider("Health Care", 1, 10, 5),
    "Internet Speed": st.slider("Internet Speed", 1, 10, 5),
    "Temperature": st.slider("Temperature", 1, 10, 5)
}
st.write("")

# User selects qualitative choices for each feature
st.subheader("2️⃣ Select your Preferred Country Temperature Level")

# Defining the qualitative options
temperature_options = ["Cold", "Moderate", "Hot"]

# Creating qualitative slider
selected_temperature = st.select_slider("Temperature Range", options=temperature_options)

# Keeping the choice in a dict format
choices = {"Temperature Range": selected_temperature}
st.write(f"Selected Temperature: {selected_temperature}")
st.write("")

# Submit button to trigger API call
if st.button("🎯 Find My Ideal Country"):
    # Prepare the data to send to the API
    user_preferences = {
        "weights": weights,
        "choices": choices
    }
    # Make API call
    response = requests.post(api_local, json=user_preferences)

    if response.status_code == 200:
        results = response.json()
        st.subheader("🎯 Top Matching Countries:")
        if isinstance(results, list):
            for country in results:
                st.write(f"{country['country']} - Score: {country['score']:.2f}")
        else:
            st.write("API response error:", results)
    else:
        st.error("No countries found. Please try again.")
