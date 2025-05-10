"""
FastAPI backend for the Ideal Country Selector application.

This module implements the API endpoints for the Ideal Country Selector application.
It provides endpoints for recommending countries based on user preferences and bridges the integration between the frontend and the recommendation algorithm.

Endpoints:
    - GET /: Simple health check endpoint
    - POST /recommend-countries: Main endpoint for generating country recommendations
"""

from fastapi import FastAPI, HTTPException
from package_folder.data_utils import load_pipeline
from package_folder.input_processor import transform_user_inputs
from package_folder.similarity import find_similar_countries
from typing import Dict, Any, List
import pandas as pd
import os
import numpy as np

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()

# Load pipeline once when the API starts - handles caching internally
pipeline = load_pipeline()

@app.get('/')
def root():
    """
    Health check endpoint that returns a simple message.
    
    Returns:
        dict: Simple status message
    """
    return {'status': 'ok'}

@app.post("/recommend-countries")
def recommend_countries_endpoint(user_inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Endpoint to generate country recommendations based on user preferences.

    Processes user inputs, finds similar countries using the core logic,
    and returns a ranked list with detailed scores and values.

    Args:
        user_inputs: Dictionary from the frontend request body containing preferences
                     (e.g., climate_preference), importances (e.g., climate_importance),
                     optional max_monthly_budget, and optional continent_preference (e.g., "EU").

    Returns:
        List of dictionaries, each representing a recommended country.
    """
    try:
        # 1. Transform raw user inputs using the input_processor module
        # where scaled_preferences is the values for the features of the ideal country, 
        # scaled_weights is the importance given to each feature, and scaled_budget is the maximum monthly budget normalized like we do for cost_of_living.
        scaled_preferences, scaled_weights, scaled_budget = transform_user_inputs(user_inputs, pipeline)

        # 2. Extract continent preference (if any)
        continent_code = user_inputs.get("continent_preference") # e.g., "EU", "AS", or None

        # 3. define the number of country recommendations
        n_neighbors = 5 # Default number of country reccomendations
        
        # 4. Find similar countries using the similarity module
        result_df,weighted_squared_deltas = find_similar_countries(
            scaled_preferences=scaled_preferences,
            scaled_weights=scaled_weights,
            scaled_budget=scaled_budget,
            continent_code=continent_code,
            n_neighbors=n_neighbors
        )

        # 5. Convert results DataFrame to list of dictionaries for JSON response
        # Handle potential NaN/NaT values appropriately for JSON serialization
        # Using replace with np.nan first, then fillna for robust conversion
        result_df = result_df.replace({pd.NaT: None, np.nan: None})
        # Ensure numerical columns are appropriate types before dict conversion
        # (Example: convert scores/deltas explicitly if needed)
        
        results = result_df.to_dict(orient="records")

        return results

    except FileNotFoundError as e:
        # Handle errors related to loading data/pipeline files
        print(f"API Error: Data or pipeline file not found. {e}")
        raise HTTPException(status_code=500, detail="Internal server error: Configuration file missing.")
    except ValueError as e:
        # Handle errors like missing columns or invalid preference types
        print(f"API Error: Value error during processing. {e}")
        # Provide a more specific error if possible, otherwise a general one
        raise HTTPException(status_code=400, detail=f"Bad request: {e}")
    except Exception as e:
        # Catch-all for other unexpected errors
        print(f"API Error: An unexpected error occurred. {e}")
        # Optionally log the full traceback here
        raise HTTPException(status_code=500, detail="Internal server error.")