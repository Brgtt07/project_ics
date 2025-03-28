import streamlit as st
import requests

# Cloud Run API URL
API_URL = 'https://project-ics-210911899890.europe-west1.run.app/predict'

st.set_page_config(page_title="Ideal Country Finder", layout="centered")
st.title("🌍 Find Your Ideal Country to Live!")
st.markdown("Customize your preferences and discover which countries suit you best!")

# --- Step 1: Preference order ---
st.header("1️⃣ Prioritize your preferences")
preference_labels = ["Climate", "Cost of Living", "Safety", "Internet Quality"]

cols = st.columns(4)
selections = []
for i, col in enumerate(cols):
    label = f"{i+1}{'st' if i == 0 else 'nd' if i == 1 else 'rd' if i == 2 else 'th'} Priority"
    options = [p for p in preference_labels if p not in selections]
    selected = col.selectbox(label, options, key=f"priority_{i}")
    selections.append(selected)

preferences_order = selections

# --- Step 2: Scoring ---
st.header("2️⃣ Rate each preference from 1 to 10")
scores = {}
for pref in preferences_order:
    key = pref.lower().replace(" ", "_")
    scores[key] = st.slider(f"{pref} Score", 1, 10, 5)

# --- Submission ---
if st.button("🎯 Find My Ideal Country"):
    data = {
        "preferences_order": preferences_order,
        **scores
    }

    # Make API call
    try:
        response = requests.get(API_URL, params=data)
        response_data = response.json()

        # Display response
        st.success("Here are your results!")
        st.json(response_data)
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        st.info("Note: Make sure the API server is running on https://project-ics-210911899890.europe-west1.run.app")
