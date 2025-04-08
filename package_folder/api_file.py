"""
FastAPI backend for the Ideal Country Selector application.

This module implements the REST API endpoints for the Ideal Country Selector application.
It provides endpoints for recommending countries based on user preferences and handles
the integration between the frontend and the recommendation algorithm.

Endpoints:
    - GET /: Simple health check endpoint
    - POST /recommend-countries: Main endpoint for generating country recommendations
"""

from fastapi import FastAPI
from package_folder.scaling_pipeline import transform_user_inputs
from package_folder.similariy_search import find_similar_countries
import pandas as pd
import os

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()

@app.get('/')
def root():
    """
    Health check endpoint that returns a simple message.
    
    Returns:
        dict: Simple status message
    """
    return {'hello': 'world'}

@app.post("/recommend-countries")
def recommend_countries(user_inputs: dict):
    """
    Generate country recommendations based on user preferences.
    
    This endpoint processes the user's preferences and importance ratings,
    filters countries based on the user's maximum monthly budget (if provided),
    the continent preference (if provided), and returns a ranked list of
    the most similar countries with detailed explanation of differences.
    
    Args:
        user_inputs: Dictionary containing user preferences and importance ratings:
            - climate_preference: String ("hot", "mild", "cold")
            - *_importance: Float (0-10) for each preference's importance
            - max_monthly_budget: Optional Float representing maximum budget in USD
            - continent_preference: Optional String representing preferred continent
    
    Returns:
        List of dictionaries, each containing:
            - country: Country name
            - similarity_score: Overall similarity score (higher is better)
            - feature specific deltas in original units (e.g., average_monthly_cost_$_delta)
            - original feature values for each country
    """
    # Process user inputs (encoding and normalization), returns a dictionary
    processed_inputs = transform_user_inputs(user_inputs)
    
    # Find similar countries using KNN with explainable results
    # The filtering for max_monthly_budget and continent is now handled inside find_similar_countries
    n_neighbors = 10  # Default number of recommendations to return
    if user_inputs.get("num_recommendations"):
        n_neighbors = min(int(user_inputs["num_recommendations"]), 50)  # Limit to maximum 50
        
    result_df = find_similar_countries(processed_inputs, n_neighbors=n_neighbors)
    
    # Return the results as a list of dictionaries
    return result_df.to_dict(orient="records")
