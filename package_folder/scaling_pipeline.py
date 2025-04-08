"""
Module for processing user inputs in the Ideal Country Selector application.

This module handles the transformation of user input preferences into normalized values 
that can be used by the recommendation algorithm. It includes functions for encoding 
qualitative preferences (like "hot" or "cold" for climate) into numerical values, and 
scaling those values to match the dataset's scale.

Input data:
    - User preferences for climate, cost of living, healthcare, safety, and internet speed
    - User importance ratings for each preference
    - Optional maximum monthly budget
    
Output data:
    - Normalized preference values (scaled between 0-1)
    - Normalized importance weights (scaled between 0-1)
    - Normalized maximum monthly budget (if provided)
"""

import numpy as np
from typing import Dict, Any
import pickle
import pandas as pd
import os
import pycountry
import pycountry_convert as pc   # pycountry_convert for country-to-continent mapping

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_country_continent_mapping():
    """
    Create a dictionary mapping country names (lowercase) to their respective continents.
    Uses pycountry_convert to map ISO country codes to continent codes.
    """
    country_to_continent = {}
    for country in pycountry.countries:
        try:
            country_alpha2 = country.alpha_2  # Get the 2-letter country code
            continent_code = pc.country_alpha2_to_continent_code(country_alpha2)  # Get the continent code
            country_to_continent[country.name.lower()] = continent_code  # Store mapping in lowercase
        except KeyError:
            continue  # Skip if no continent found
    return country_to_continent

# Load country-to-continent mapping once
country_to_continent = get_country_continent_mapping()


def transform_user_inputs(user_input_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Transform user input dictionary received from the frontend into normalized values suitable for model prediction.
    
    This function processes the raw user input from the frontend form, converting categorical 
    preferences into numerical values and scaling them to match the dataset's scale using a 
    pre-fitted scaling pipeline. It also scales importance ratings and handles the maximum 
    monthly budget if provided.
    
    Args:
        user_input_dict: Dictionary with user preferences and importance ratings. May include:
            - climate_preference: String ("hot", "mild", "cold")
            - cost_of_living_preference: String ("low", "moderate", "high")
            - healthcare_preference: String ("excellent", "good", "fair")
            - safety_preference: String ("very_safe", "safe", "moderate")
            - internet_speed_preference: String ("fast", "moderate", "slow")
            - *_importance: Float (0-10) for each preference's importance
            - max_monthly_budget: Optional float representing maximum budget in USD

    Returns:
        Dictionary with normalized preference values and importance weights:
            - Each preference value is scaled between 0-1
            - Each importance value is scaled between 0-1
            - max_monthly_budget is scaled if provided
    """
    normalized_inputs = {}

    # Step 1: Transform categorical preferences to numerical values
    # Mapping each preference using the helper function. We  use get() with a default value for each preference in case the key is missing
    def assign_preference(key, default_if_no_value):
        if key in user_input_dict:
            return encode_preference(user_input_dict[key], key)
        return default_if_no_value

    normalized_inputs["climate_preference"] = assign_preference("climate_preference", default_if_no_value=21.0)
    normalized_inputs["cost_of_living_preference"] = assign_preference("cost_of_living_preference", default_if_no_value=579.8)
    normalized_inputs["healthcare_preference"] = assign_preference("healthcare_preference", default_if_no_value=86.5)
    normalized_inputs["safety_preference"] = assign_preference("safety_preference", default_if_no_value=84.7)
    normalized_inputs["internet_speed_preference"] = assign_preference("internet_speed_preference", default_if_no_value=345.3)

    # Step 2: Apply transformation pipeline to all preference values
    # This will apply pre-fitted min-max scaling to the preference values
    # the pipeline is fitted on the "scaled_merged_data_after_imputation.csv" dataset, not sure if it expects specific column names


    # Import the pre-fitted pipeline and apply the transformation
    pipeline_path = os.path.join(project_dir, 'models', 'scaling_pipeline.pkl')
    with open(pipeline_path, 'rb') as f:
        pipe = pickle.load(f)

    preferences_to_transform = pd.DataFrame([{
        "average_monthly_cost_$": normalized_inputs["cost_of_living_preference"],
        "average_yearly_temperature": normalized_inputs["climate_preference"],
        "internet_speed_mbps": normalized_inputs["internet_speed_preference"],
        "safety_index": normalized_inputs["safety_preference"],
        "Healthcare Index": normalized_inputs["healthcare_preference"]
    }])

    transformed_preferences = pipe.transform(preferences_to_transform)

    # Update normalized inputs with transformed values
    normalized_inputs["cost_of_living_preference"] = float(transformed_preferences[0][0])
    normalized_inputs["climate_preference"] = float(transformed_preferences[0][1])
    normalized_inputs["internet_speed_preference"] = float(transformed_preferences[0][2])
    normalized_inputs["safety_preference"] = float(transformed_preferences[0][3])
    normalized_inputs["healthcare_preference"] = float(transformed_preferences[0][4])

    # Step 3: Scale importance values between 0 and 1
    importance_keys = [k for k in user_input_dict.keys() if k.endswith("_importance")]
    for key in importance_keys:
        normalized_inputs[key] = user_input_dict[key] / 10.0  # Scale importance values

    # Step 4: Handle max_monthly_budget (need to also scale this value, and to scale it we need to import the weights from the pipeline, complicated)
    if user_input_dict.get("max_monthly_budget") is not None:
        # Create a single-row DataFrame with the budget and dummy values for other columns
        simulated_df = pd.DataFrame([{
            'average_monthly_cost_$': user_input_dict["max_monthly_budget"],
            'average_yearly_temperature': 0,
            'internet_speed_mbps': 0,
            'safety_index': 0,
            'Healthcare Index': 0
        }])

        #access the column transformer from the pipeline
        column_transformer = pipe.named_steps['column_transformer']

        # Transform using the column transformer. [0][0] because the transform returns a numpy array
        normalized_inputs["max_monthly_budget"] = column_transformer.transform(simulated_df)[0][0]

    # Filter by continent (if selected by user)
    if "continent_preference" in user_input_dict:
        selected_continent = user_input_dict["continent_preference"]  # e.g., "EU"
        filtered_countries = [country for country, continent in country_to_continent.items() if continent == selected_continent]
        normalized_inputs["filtered_countries"] = [c.lower() for c in filtered_countries]  # Store in lowercase for dataset matching

    return normalized_inputs


def encode_preference(preference: any, preference_type: str) -> float:
    """
    Maps a qualitative user preference to a numerical value based on its type.
    
    This function converts categorical user preferences (e.g., "hot", "moderate", "excellent") 
    into numerical values that can be processed by the algorithm. Each preference type has 
    its own mapping scale based on the real-world metrics in the dataset.
    
    Args:
        preference: The qualitative user preference (e.g., "hot", "low", "excellent")
        preference_type: The type of preference to encode, one of:
            - "climate_preference" - values map to average yearly temperature in °C
            - "cost_of_living_preference" - values map to average monthly cost in USD
            - "healthcare_preference" - values map to healthcare index (0-100)
            - "safety_preference" - values map to safety index (0-100)
            - "internet_speed_preference" - values map to internet speed in Mbps

    Returns:
        A float value representing the encoded preference, suitable for further scaling
        
    Raises:
        ValueError: If an invalid preference or preference_type is provided
    """
    if preference_type == "climate_preference":
        if preference == "hot":
            return 25.0
        elif preference == "mild":
            return 18.0
        elif preference == "cold":
            return 11.0
        else:
            raise ValueError(f"Invalid climate preference: '{preference}'. Expected one of: 'hot', 'mild', 'cold'")

    elif preference_type == "cost_of_living_preference":
        if preference == "low":
            return 400.0
        elif preference == "moderate":
            return 800.0
        elif preference == "high":
            return 2500.0
        else:
            raise ValueError(f"Invalid cost of living preference: '{preference}'. Expected one of: 'low', 'moderate', 'high'")

    elif preference_type == "healthcare_preference":
        if preference == "excellent":
            return 75.0
        elif preference == "good":
            return 65.0
        elif preference == "fair":
            return 50.0
        else:
            raise ValueError(f"Invalid healthcare preference: '{preference}'. Expected one of: 'excellent', 'good', 'fair'")

    elif preference_type == "safety_preference":
        if preference == "very_safe":
            return 75.0
        elif preference == "safe":
            return 65.0
        elif preference == "moderate":
            return 50.0
        else:
            raise ValueError(f"Invalid safety preference: '{preference}'. Expected one of: 'very_safe', 'safe', 'moderate'")

    elif preference_type == "internet_speed_preference":
        if preference == "fast":
            return 200.0
        elif preference == "moderate":
            return 100.0
        elif preference == "slow":
            return 50.0
        else:
            raise ValueError(f"Invalid internet speed preference: '{preference}'. Expected one of: 'fast', 'moderate', 'slow'")
    else:
        raise ValueError(f"Invalid preference type: '{preference_type}'. Did you mean 'climate', 'cost_of_living', 'healthcare', 'safety', or 'internet_speed'?")
    return None  # Fallback for any unknown preference types
