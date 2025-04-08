import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any

# Import necessary components from data_utils
from .data_utils import load_pipeline

# --- Preference Encoding ---

def encode_preference(preference: str, preference_type: str) -> float:
    """Maps qualitative user preference to a numerical value based on dataset metrics."""
    # Mappings based on analysis of original data/user expectations
    mappings = {
        "climate_preference": {"hot": 25.0, "mild": 18.0, "cold": 11.0},
        # Placeholders - ensure these map meaningfully if sliders are added later
        "cost_of_living_preference": {"low": 400.0, "moderate": 800.0, "high": 2500.0},
        "healthcare_preference": {"excellent": 75.0, "good": 65.0, "fair": 50.0},
        "safety_preference": {"very_safe": 75.0, "safe": 65.0, "moderate": 50.0},
        "internet_speed_preference": {"fast": 200.0, "moderate": 100.0, "slow": 50.0}
    }
    default_values = {
        "climate_preference": 18.0, # mild
        "cost_of_living_preference": 800.0, # moderate (default if not provided)
        "healthcare_preference": 65.0, # good (default if not provided)
        "safety_preference": 65.0, # safe (default if not provided)
        "internet_speed_preference": 100.0 # moderate (default if not provided)
    }

    # Normalize preference string (lowercase, handle potential whitespace)
    preference = str(preference).lower().strip()

    if preference_type not in mappings:
        raise ValueError(f"Invalid preference type: '{preference_type}'")

    # Use the mapping, falling back to default if the specific preference isn't recognized
    if preference not in mappings[preference_type]:
         print(f"Warning: Invalid preference value '{preference}' for {preference_type}. Using default.")
         return default_values[preference_type]

    return mappings[preference_type][preference]


# --- Input Transformation ---

def transform_user_inputs(user_input_dict: Dict[str, Any], pipeline: Any) -> Tuple[Dict[str, float], Dict[str, float], Optional[float]]:
    """
    Transforms raw user inputs from the frontend into scaled preferences, weights, and budget.

    Args:
        user_input_dict: Raw inputs from the frontend.
        pipeline: The loaded scaling pipeline (passed in for dependency injection).

    Returns:
        Tuple containing:
        - Dictionary of scaled preference values (0-1) keyed by feature name.
        - Dictionary of scaled importance weights (0-1) keyed by feature name.
        - Scaled maximum monthly budget (0-1), or None.
    """
    scaled_preferences = {}
    scaled_weights = {}
    scaled_budget = None

    # Feature names used in the pipeline/dataset
    pipeline_features = [
        "average_monthly_cost_$",
        "average_yearly_temperature",
        "internet_speed_mbps",
        "safety_index",
        "Healthcare Index"
    ]

    # 1. Encode qualitative preferences to numerical values
    # Use get with defaults for preferences that might be missing or have non-slider inputs
    numerical_prefs = {
        "average_yearly_temperature": encode_preference(
            user_input_dict.get("climate_preference", "mild"), "climate_preference"
        ),
        # For factors without sliders, use defaults defined in encode_preference
        "average_monthly_cost_$": encode_preference(
             user_input_dict.get("cost_of_living_preference", "moderate"), "cost_of_living_preference"
        ),
        "Healthcare Index": encode_preference(
            user_input_dict.get("healthcare_preference", "good"), "healthcare_preference"
        ),
        "safety_index": encode_preference(
            user_input_dict.get("safety_preference", "safe"), "safety_preference"
        ),
        "internet_speed_mbps": encode_preference(
            user_input_dict.get("internet_speed_preference", "moderate"), "internet_speed_preference"
        )
    }

    # 2. Scale numerical preferences using the pipeline
    # Ensure the DataFrame has columns in the exact order the pipeline expects
    prefs_df = pd.DataFrame([numerical_prefs], columns=pipeline_features)
    transformed_prefs = pipeline.transform(prefs_df)[0] # Get the first (only) row

    # Map transformed values back to dictionary keyed by feature name
    for i, feature_name in enumerate(pipeline_features):
        scaled_preferences[feature_name] = float(transformed_prefs[i])

    # 3. Scale importance weights (0-10 -> 0-1)
    importance_map = {
        "climate_importance": "average_yearly_temperature",
        "cost_of_living_importance": "average_monthly_cost_$",
        "healthcare_importance": "Healthcare Index",
        "safety_importance": "safety_index",
        "internet_speed_importance": "internet_speed_mbps"
    }
    for key, feature_name in importance_map.items():
        # Use default importance of 5 if not provided, ensure conversion to float
        importance = float(user_input_dict.get(key, 5.0))
        scaled_weights[feature_name] = max(0.0, min(1.0, importance / 10.0)) # Clamp between 0 and 1

    # 4. Scale max_monthly_budget if provided
    budget_input = user_input_dict.get("max_monthly_budget")
    if budget_input is not None:
        try:
            budget = float(budget_input)
            if budget > 0:
                # Create DataFrame with budget and dummy values for scaling
                # Ensure columns match the order expected by the transformer
                budget_df = pd.DataFrame([{ 
                    feat: budget if feat == 'average_monthly_cost_$' else 0 
                    for feat in pipeline_features
                }], columns=pipeline_features)
                
                # Use the specific column transformer for consistency
                column_transformer = pipeline.named_steps['column_transformer']
                scaled_budget = column_transformer.transform(budget_df)[0][0] # Budget is the first column
        except (ValueError, TypeError):
            print(f"Warning: Invalid budget value received: {budget_input}. Ignoring budget.")
            scaled_budget = None

    return scaled_preferences, scaled_weights, scaled_budget 