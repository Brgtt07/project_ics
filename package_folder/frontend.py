import streamlit as st
import requests

# Cloud Run API URL
API_URL = 'https://project-ics-210911899890.europe-west1.run.app//recommend-countries'
# Local API URL for testing
api_local = 'http://localhost:8000//recommend-countries'

st.title("🌍 Find Your Ideal Country to Live!")
st.write("")

# Initialize the user inputs dictionary
user_inputs = {}

# Climate/Temperature
st.subheader("🌡️ Temperature Preference")
# Defining the qualitative options
temperature_options = ["Cold", "Moderate", "Hot"]
user_inputs["climate_preference"] = st.select_slider("Temperature Range", options=temperature_options).lower()
user_inputs["climate_importance"] = st.slider("How important is climate to you?", 0, 10, 5)
st.write("")

# Cost of Living
st.subheader("💰 Cost of Living")
#commented options but left them in case we want to use them in the future
    #cost_options = ["Low", "Moderate", "High"]
    #user_inputs["cost_of_living_preference"] = st.select_slider("Cost of Living Range", options=cost_options).lower()
user_inputs["cost_of_living_importance"] = st.slider("How important is cost of living to you?", 0, 10, 5)
user_inputs["max_monthly_budget"] = st.number_input("Optional: Maximum monthly budget (USD)", min_value=0, value=0, step=100)
if user_inputs["max_monthly_budget"] == 0:
    user_inputs["max_monthly_budget"] = None
st.write("")

# Healthcare
st.subheader("🏥 Healthcare")
#commented options but left them in case we want to use them in the future
    #healthcare_options = ["Fair", "Good", "Excellent"]
    #user_inputs["healthcare_preference"] = st.select_slider("Healthcare Quality", options=healthcare_options).lower()
user_inputs["healthcare_importance"] = st.slider("How important is healthcare to you?", 0, 10, 5)
st.write("")

# Safety
st.subheader("🛡️ Safety")
#commented options but left them in case we want to use them in the future
    #safety_options = ["Moderate", "Safe", "Very Safe"]
    #user_inputs["safety_preference"] = st.select_slider("Safety Level", options=safety_options).lower()
user_inputs["safety_importance"] = st.slider("How important is safety to you?", 0, 10, 5)
st.write("")

# Internet Speed
st.subheader("🌐 Internet Speed")
#commented options but left them in case we want to use them in the future
    #internet_options = ["Slow", "Moderate", "Fast"]
    #user_inputs["internet_speed_preference"] = st.select_slider("Internet Speed", options=internet_options).lower()
user_inputs["internet_speed_importance"] = st.slider("How important is internet speed to you?", 0, 10, 5)
st.write("")

# Submit button to trigger API call
if st.button("🎯 Find My Ideal Country"):
    # Make API call
    response = requests.get(api_local, params=user_inputs)

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

#if st.button("🧪 Test User Input"):
        # Make API call to the test endpoint
        #response = requests.post("http://localhost:8000/test_user_input", json=user_inputs)

        #if response.status_code == 200:
        #    test_results = response.json()
        #    st.subheader("🧪 Test User Input Response:")
        #    st.write(test_results)
        #else:
        #    st.error("Error testing user input. Please try again.")
