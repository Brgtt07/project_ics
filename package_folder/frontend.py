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
import pandas as pd

# Cloud Run API URL
API_URL = 'https://project-ics-210911899890.europe-west1.run.app/recommend-countries'
# Local API URL for testing
api_local = 'http://localhost:8000/recommend-countries'

st.title("🌍 Find Your Ideal Country to Live!")
st.write("")

# Continent selection
st.subheader("🗺️📍 Continent Preference")
continent_options = {
    "Any": None,
    "Africa": "AF",
    "Asia": "AS",
    "Europe": "EU",
    "North America": "NA",
    "Oceania": "OC",
    "South America": "SA"
}
continent_preference = st.selectbox("Select Continent", options=list(continent_options.keys()))
selected_continent = continent_options[continent_preference]
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
        'internet_speed_importance': internet_speed_importance,
        'continent_preference': selected_continent
    }

    if max_monthly_budget is not None:
        data['max_monthly_budget'] = max_monthly_budget

    # Make API call with POST request
    response = requests.post(api_local, json=data)
    st.markdown("---")  # Adds a horizontal line for separation
    if response.status_code == 200:
        results = response.json()
        st.subheader("🎯 Top Matching Countries")
        
        if isinstance(results, list) and len(results) > 0:
            # Identify the score key used in the API response
            sample_country = results[0]
            # Different versions might use different keys for the overall score
            if 'country_score' in sample_country:
                score_key = 'country_score'
            elif 'similarity_score' in sample_country:
                score_key = 'similarity_score'
            else:
                # If we can't find either key, use the first key that contains 'score'
                score_candidates = [k for k in sample_country.keys() if 'score' in k.lower()]
                score_key = score_candidates[0] if score_candidates else 'country'  # Fallback if no score found
            
            for i, country in enumerate(results, 1):
                try:
                    # Get country name with fallback
                    country_name = country.get('country', f'Country {i}')
                    if isinstance(country_name, str):
                        country_name = country_name.title()
                    
                    # Get score with fallback
                    if score_key in country and country[score_key] is not None:
                        country_score = country[score_key] * 100
                    else:
                        country_score = 0
                    
                    # Create an expander for each country to make the list more compact
                    with st.expander(f"#{i} 🏆 {country_name} - Overall Match: {country_score:.1f}%", expanded=i<=3):
                        # Display overall score at the top with a larger progress bar
                        st.markdown("Overall Match Score")
                        st.progress(float(country[score_key]) if score_key in country else 0)
                        
                        st.markdown("Individual Factors")
                        # Create columns for individual feature scores
                        col1, col2 = st.columns(2)
                        
                        # Display individual feature match scores in a more compact layout
                        feature_map = {
                            'average_monthly_cost_$': '💰 Cost of Living',
                            'average_yearly_temperature': '🌡️ Temperature',
                            'internet_speed_mbps': '🌐 Internet Speed',
                            'safety_index': '🛡️ Safety',
                            'Healthcare Index': '🏥 Healthcare'
                        }
                        
                        # Alternative names that might appear in the API response
                        feature_aliases = {
                            'average_monthly_cost_$': ['cost_of_living', 'cost', 'monthly_cost'],
                            'average_yearly_temperature': ['temperature', 'climate', 'yearly_temperature'],
                            'internet_speed_mbps': ['internet', 'internet_speed'],
                            'safety_index': ['safety'],
                            'Healthcare Index': ['healthcare', 'health']
                        }
                        
                        # Map feature names to their potential match score keys with more fallbacks
                        feature_score_map = {}
                        for feature, label in feature_map.items():
                            # Try different naming patterns for match scores
                            potential_keys = [
                                f"{feature}_match_score",
                                f"{feature}_score"
                            ]
                            
                            # Add alias-based keys
                            if feature in feature_aliases:
                                for alias in feature_aliases[feature]:
                                    potential_keys.extend([
                                        f"{alias}_match_score",
                                        f"{alias}_score"
                                    ])
                            
                            # Look for any key containing feature name or alias and 'score'
                            score_keys = [k for k in country.keys() if 'score' in k.lower()]
                            for key in score_keys:
                                if feature.lower() in key.lower():
                                    potential_keys.append(key)
                                elif feature in feature_aliases:
                                    for alias in feature_aliases[feature]:
                                        if alias.lower() in key.lower():
                                            potential_keys.append(key)
                            
                            # Find the first matching key that exists
                            for key in potential_keys:
                                if key in country:
                                    feature_score_map[feature] = key
                                    break
                        
                        # If no feature scores found, try to calculate them from deltas if available
                        if not feature_score_map:
                            for feature in feature_map.keys():
                                delta_key = f"{feature}_delta"
                                if delta_key in country and not pd.isna(country[delta_key]):
                                    # Create a simple score from the delta (smaller delta = higher score)
                                    max_delta = 1.0  # Assume a reasonable max delta
                                    delta = abs(float(country[delta_key]))
                                    score = max(0, 1.0 - (delta / max_delta))
                                    # Store the score directly in the country dict
                                    match_key = f"{feature}_match_score"
                                    country[match_key] = score
                                    feature_score_map[feature] = match_key
                        
                        # If still no feature scores found and no warning shown yet, show a message
                        if not feature_score_map:
                            st.info("Detailed factor match scores not available. Try adjusting your preferences or running the search again.")
                        
                        # Left column features
                        with col1:
                            for idx, (feature, label) in enumerate(list(feature_map.items())[:3]):
                                if feature in feature_score_map:
                                    match_key = feature_score_map[feature]
                                    if match_key in country and not pd.isna(country[match_key]):
                                        score = country[match_key] * 100
                                        # Create a color-coded progress bar based on the score
                                        st.text(f"{label}: {score:.1f}%")
                                        st.progress(float(country[match_key]))
                        
                        # Right column features
                        with col2:
                            for idx, (feature, label) in enumerate(list(feature_map.items())[3:]):
                                if feature in feature_score_map:
                                    match_key = feature_score_map[feature]
                                    if match_key in country and not pd.isna(country[match_key]):
                                        score = country[match_key] * 100
                                        # Create a color-coded progress bar based on the score
                                        st.text(f"{label}: {score:.1f}%")
                                        st.progress(float(country[match_key]))
                except Exception as e:
                    st.error(f"Error displaying country #{i}: {str(e)}")
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
