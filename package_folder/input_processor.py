import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any

# Import necessary components from data_utils
from .data_utils import load_pipeline

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

    # 1. Map preferences to numerical values
    # Climate preference mapping
    climate_mapping = {"hot": 25.0, "mild": 18.0, "cold": 11.0}
    climate_default = 18.0  # mild
    
    # Get climate preference and normalize it
    climate_pref = str(user_input_dict.get("climate_preference", "mild")).lower().strip()
    
    # Validate climate preference
    if climate_pref not in climate_mapping:
        print(f"Warning: Invalid climate preference '{climate_pref}'. Using default (mild).")
        climate_value = climate_default
    else:
        climate_value = climate_mapping[climate_pref]
    
    # Create dictionary of numerical preference values
    numerical_prefs = {
        # Use default values for preferences that don't have a slider in the frontend
        "average_monthly_cost_$": 800.0,
        "average_yearly_temperature": climate_value,  
        "internet_speed_mbps": 100.0,
        "safety_index": 65.0,
        "Healthcare Index": 65.0
    }

    # 2. Scale numerical preferences using the pipeline
    # ATTENTION!!! Ensure the DataFrame has columns in the exact order the pipeline expects
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