import numpy as np
from typing import Dict, Any
import pickle
import pandas as pd
import os

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def transform_user_inputs(user_input_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Transform user input dictionary recieved from the frontend into normalized values suitable for model / simple algorithm prediction.
    
    Args:
        user_input_dict: Dictionary with user preferences and importance ratings
        
    Returns:
        Dictionary with normalized preference values and importance weights
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
    #if user_input_dict.get("max_monthly_budget") is not None:
    #    normalized_inputs["max_monthly_budget"] = user_input_dict["max_monthly_budget"]
    return normalized_inputs   



def encode_preference(preference: any, preference_type: str) -> float:
    """
    Maps a user preference to a normalized value based on its type.
    
    Args:
        preference: The qualitative user preference e.g. "hot"
        preference_type: The type of preference the "preference" arg refers to (e.g., "climate", "cost_of_living", etc.)
        
    Returns:
        An arbitrary float value to encode the qualitative preference into a number.
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




