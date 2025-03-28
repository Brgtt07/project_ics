import streamlit as st
import requests

# Cloud Run API URL
API_URL = 'https://project-ics-210911899890.europe-west1.run.app/predict'
api_local = 'http://localhost:8000/predict'

st.title("🌍 Find Your Ideal Country to Live!")

# User selects the importance of each feature (1-10 scale)
st.subheader("1️⃣ Rate the Importance of Each Category (1 = Not Important, 10 = Very Important)")
weights = {
    "Safety": st.slider("Safety", 1, 10, 5),
    "Cost of Living": st.slider("Cost of Living", 1, 10, 5),
    "Health Care Index": st.slider("Health Care", 1, 10, 5),
    "Internet Speed": st.slider("Internet Speed", 1, 10, 5),
    "Temperature": st.slider("Temperature", 1, 10, 5)
}

# User selects qualitative choices for each feature
st.subheader("2️⃣ Select your Preferred Quality Level for Each Category")
choices = {
    "Safety": st.selectbox("Safety", ["Very Low", "Low", "Moderate", "High", "Very High"]),
    "Cost of Living": st.selectbox("Cost of Living", ["Very Low", "Low", "Moderate", "High", "Very High"]),
    "Health Care Index": st.selectbox("Health Care", ["Very Low", "Low", "Moderate", "High", "Very High"]),
    "Internet Speed": st.selectbox("Internet Speed", ["Very Low", "Low", "Moderate", "High", "Very High"]),
    "Temperature": st.selectbox("Temperature", ["Very Low", "Low", "Moderate", "High", "Very High"])
}

# User selects number of countries to receive as output
num_countries = st.slider("How many countries do you want to see?", 1, 10, 5)

if st.button("🎯 Find My Ideal Country"):
    # Prepare the data to send to the API
    user_preferences = {
        "weights": {
            "Safety": weights["Safety"],
            "Cost of Living": weights["Cost of Living"],
            "Health Care Index": weights["Health Care Index"],
            "Internet Speed": weights["Internet Speed"],
            "Temperature": weights["Temperature"]
        },
        "choices": {
            "Safety": choices["Safety"],
            "Cost of Living": choices["Cost of Living"],
            "Health Care Index": choices["Health Care Index"],
            "Internet Speed": choices["Internet Speed"],
            "Temperature": choices["Temperature"]
        },
        "num_countries": num_countries
    }

    response = requests.post(api_local, json=user_preferences)

    if response.status_code == 200:
        results = response.json()
        st.subheader("Top Matching Countries:")
        if isinstance(results, list):
            for country in results:
                st.write(f"{country['country']} - Score: {country['score']:.2f}")
        else:
            st.write("API response error:", results)
    else:
        st.error("No countries found. Please try again.")
