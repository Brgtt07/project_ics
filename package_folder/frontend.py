"""
Streamlit frontend for the Ideal Country Selector application.

This module implements the user interface for the Ideal Country Selector application
using Streamlit. It provides input forms for users to specify their preferences and
importance ratings for various factors (climate, cost of living, healthcare, safety,
and internet speed). The user inputs are sent to the backend API, which returns 
recommended countries based on the user's preferences.

The frontend displays:
    - Input sliders and selectors for user preferences
    - Importance rating sliders for each factor
    - Optional maximum monthly budget input
    - Results section showing the top matching countries with scores
"""

import streamlit as st
import requests

# Cloud Run API URL
API_URL = 'https://project-ics-210911899890.europe-west1.run.app/recommend-countries'
# Local API URL for testing
api_local = 'http://localhost:8000/recommend-countries'

st.title("🌍 Find Your Ideal Country to Live!")
st.write("")

# Climate/Temperature
st.subheader("🌡️ Temperature Preference")
# Defining the qualitative options
temperature_options = ["Cold", "Mild", "Hot"]
climate_preference = st.select_slider("Temperature Range", options=temperature_options).lower()
climate_importance = st.slider("How important is climate to you?", 0, 10, 5)
st.write("")

# Cost of Living
st.subheader("💰 Cost of Living")
#commented options but left them in case we want to use them in the future
    #cost_options = ["Low", "Moderate", "High"]
    #cost_of_living_preference = st.select_slider("Cost of Living Range", options=cost_options).lower()
cost_of_living_importance = st.slider("How important is cost of living to you?", 0, 10, 5)
max_monthly_budget = st.number_input("Optional: Maximum monthly budget (USD)", min_value=0, value=0, step=100)
if max_monthly_budget == 0:
    max_monthly_budget = None
st.write("")

# Healthcare
st.subheader("🏥 Healthcare")
#commented options but left them in case we want to use them in the future
    #healthcare_options = ["Fair", "Good", "Excellent"]
    #healthcare_preference = st.select_slider("Healthcare Quality", options=healthcare_options).lower()
healthcare_importance = st.slider("How important is healthcare to you?", 0, 10, 5)
st.write("")

# Safety
st.subheader("🛡️ Safety")
#commented options but left them in case we want to use them in the future
    #safety_options = ["Moderate", "Safe", "Very Safe"]
    #safety_preference = st.select_slider("Safety Level", options=safety_options).lower()
safety_importance = st.slider("How important is safety to you?", 0, 10, 5)
st.write("")

# Internet Speed
st.subheader("🌐 Internet Speed")
#commented options but left them in case we want to use them in the future
    #internet_options = ["Slow", "Moderate", "Fast"]
    #internet_speed_preference = st.select_slider("Internet Speed", options=internet_options).lower()
internet_speed_importance = st.slider("How important is internet speed to you?", 0, 10, 5)
st.write("")

# Submit button to trigger API call
if st.button("🎯 Find My Ideal Country"):
    # Prepare the data
    data = {
        'climate_preference': climate_preference,
        'climate_importance': climate_importance,
        'cost_of_living_importance': cost_of_living_importance,
        'healthcare_importance': healthcare_importance,
        'safety_importance': safety_importance,
        'internet_speed_importance': internet_speed_importance
    }
    if max_monthly_budget is not None:
        data['max_monthly_budget'] = max_monthly_budget

    # Make API call with POST request
    response = requests.post(api_local, json=data)
    st.markdown("---")  # Adds a horizontal line for separation
    if response.status_code == 200:
        results = response.json()
        st.subheader("🎯 Top Matching Countries:")
        if isinstance(results, list):
            for i, country in enumerate(results, 1):
                # Display country name and score in a single line
                st.write(f"**#{i} 🏆 {country['country']}** - Match Score: {country['country_score'] * 100:.1f}")
        else:
            st.write("API response error:", results)
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

#if st.button("🧪 Test User Input"):
    # Make API call to the test endpoint
    #test_data = {
        #'climate_preference': climate_preference,
        #'climate_importance': climate_importance,
        #'cost_of_living_importance': cost_of_living_importance,
        #'healthcare_importance': healthcare_importance,
        #'safety_importance': safety_importance,
        #'internet_speed_importance': internet_speed_importance
    #}
    #response = requests.post("http://localhost:8000/test_user_input", json=test_data)

    #if response.status_code == 200:
        #test_results = response.json()
        #st.subheader("🧪 Test User Input Response:")
        #st.write(test_results)
    #else:
        #st.error("Error testing user input. Please try again.")
