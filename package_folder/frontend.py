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
