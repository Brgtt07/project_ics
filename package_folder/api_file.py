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
from package_folder.weighted_sum import weighted_sum
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
    and returns a ranked list of the top matching countries.
    
    Args:
        user_inputs: Dictionary containing user preferences and importance ratings:
            - climate_preference: String ("hot", "mild", "cold")
            - *_importance: Float (0-10) for each preference's importance
            - max_monthly_budget: Optional Float representing maximum budget in USD
    
    Returns:
        List of dictionaries, each containing:
            - country: Country name
            - country_score: Similarity score (higher is better)
    """
    # Process user inputs (encoding and normalization), returns a dictionary
    processed_inputs = transform_user_inputs(user_inputs)

    # Load the dataset
    data_path = os.path.join(project_dir, "raw_data", "merged_country_level", "scaled_merged_data_after_imputation.csv")
    data = pd.read_csv(data_path)

    #Filter the dataset based on the max_monthly_budget
    if 'max_monthly_budget' in processed_inputs:
        data = data[data['average_monthly_cost_$'] <= processed_inputs['max_monthly_budget']]

    # Calculate weighted scores
    result_df = weighted_sum(data, processed_inputs)

    return result_df.to_dict(orient="records")
