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
import os

# --- Configuration ---
# Use environment variable for API URL or fallback to local
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000") 
RECOMMEND_ENDPOINT_URL = f"{API_BASE_URL}/recommend-countries"

# --- UI Setup --- 
st.title("🌍 Find Your Dream Country to Live!")
st.write("")

# --- User Inputs ---

# Continent Selection
st.subheader("🗺️📍 Continent Preference")
continent_options = {
    "Any": None, "Africa": "AF", "Asia": "AS", "Europe": "EU",
    "North America": "NA", "Oceania": "OC", "South America": "SA"
}
continent_key = st.selectbox("Select Continent", options=list(continent_options.keys()), index=0)
selected_continent = continent_options[continent_key]
st.write("")

# Climate / Temperature
st.subheader("🌡️ Climate / Temperature")
temperature_options = ["Cold", "Mild", "Hot"]
climate_preference = st.select_slider(
    "Preferred Temperature", options=temperature_options, value="Mild"
).lower()
climate_importance = st.slider("Importance", 0, 10, 5, key="climate_importance")
st.write("")

# Cost of Living
st.subheader("💰 Cost of Living")
cost_of_living_importance = st.slider("Importance", 0, 10, 5, key="cost_importance")
max_monthly_budget = st.number_input(
    "Optional: Max Monthly Budget (in USD)", min_value=0, value=0, step=100
)
# Treat 0 as no budget specified
max_monthly_budget = None if max_monthly_budget == 0 else max_monthly_budget
st.write("")

# Healthcare
st.subheader("🏥 Healthcare")
healthcare_importance = st.slider("Importance", 0, 10, 5, key="health_importance")
st.write("")

# Safety
st.subheader("🛡️ Safety")
safety_importance = st.slider("Importance", 0, 10, 5, key="safety_importance")
st.write("")

# Internet Speed
st.subheader("🌐 Internet Speed")
internet_speed_importance = st.slider("Importance", 0, 10, 5, key="internet_importance")
st.write("")

# --- API Call and Results ---
st.markdown("---") # Separator

if st.button("🎯 Find My Ideal Country", use_container_width=True):
    # Prepare payload for API
    payload = {
        'climate_preference': climate_preference,
        'climate_importance': climate_importance,
        'cost_of_living_importance': cost_of_living_importance,
        'healthcare_importance': healthcare_importance,
        'safety_importance': safety_importance,
        'internet_speed_importance': internet_speed_importance,
        'continent_preference': selected_continent,
        'max_monthly_budget': max_monthly_budget
    }

    # Make API call
    try:
        with st.spinner("Finding your perfect match..."):
            # Use RECOMMEND_ENDPOINT_URL defined earlier
            response = requests.post(RECOMMEND_ENDPOINT_URL, json=payload, timeout=30) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        results = response.json()
        st.subheader("🏆 Top Matching Countries")

        if isinstance(results, list) and results:
            # Define feature keys expected from API and their display labels/icons
            feature_display_map = {
                'average_monthly_cost_$': '💰 Cost of Living',
                'average_yearly_temperature': '🌡️ Temperature',
                'internet_speed_mbps': '🌐 Internet Speed',
                'safety_index': '🛡️ Safety',
                'Healthcare Index': '🏥 Healthcare'
            }

            for i, country_data in enumerate(results, 1):
                country_name = country_data.get('country', f'Unknown Country {i}').title()
                # Use 'similarity_score' directly
                overall_score = country_data.get('similarity_score', 0)
                # Handle potential 'N/A' string from API conversion if fillna('N/A') was used
                if isinstance(overall_score, str) and overall_score == 'N/A':
                    overall_score = 0
                overall_score_percent = (float(overall_score) or 0) * 100

                expander_title = f"#{i} {country_name} - Overall Match: {overall_score_percent:.0f}%"
                with st.expander(expander_title, expanded=(i <= 1)): # Expand top 3
                    st.markdown("**Overall Match score:**")
                    st.progress(float(overall_score or 0))
                    st.markdown("**Detailed Scores:**")

                    # Use columns for factor scores based on the number of features
                    feature_items = list(feature_display_map.items())
                    num_features = len(feature_items)
                    cols = st.columns(num_features) # Create columns for each feature

                    for idx, (feature_key, label) in enumerate(feature_items):
                        with cols[idx]:
                            # Construct the expected match score key
                            match_score_key = f"{feature_key}_match_score"
                            feature_score = country_data.get(match_score_key, 0)
                            # Handle potential 'N/A' string
                            if isinstance(feature_score, str) and feature_score == 'N/A':
                                feature_score = 0
                            feature_score_percent = (float(feature_score) or 0) * 100

                            # Get original value and delta for display (optional, could add tooltips)
                            # original_val = country_data.get(f"{feature_key}_original", "N/A")
                            # delta = country_data.get(f"{feature_key}_delta", "N/A")
                            # tooltip_text = f"Value: {original_val} (Delta: {delta:.1f})"

                            st.text(f"{label}: {feature_score_percent:.0f}%")
                            st.progress(float(feature_score or 0))

        elif isinstance(results, list) and not results:
            st.info("No countries found matching your criteria. Try adjusting your preferences or budget.")
        else:
            # Handle unexpected response format
            st.error("Received an unexpected response from the server.")
            st.json(results) # Show the raw response for debugging

    except requests.exceptions.RequestException as e:
        st.error(f"Network error: Could not connect to the recommendation service. ({e})")
    except Exception as e:
        st.error(f"An error occurred while processing results: {e}")
