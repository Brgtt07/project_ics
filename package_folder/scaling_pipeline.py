import numpy as np
from typing import Dict, Any
import pickle

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
    normalized_inputs["climate_preference"] = encode_preference(
        user_input_dict.get("climate_preference", 20.7), "climate"
    )
    normalized_inputs["cost_of_living_preference"] = encode_preference(
        user_input_dict.get("cost_of_living_preference", 300.0), "cost_of_living"
    )
    normalized_inputs["healthcare_preference"] = encode_preference(
        user_input_dict.get("healthcare_preference", 82.78), "healthcare"
    )
    normalized_inputs["safety_preference"] = encode_preference(
        user_input_dict.get("safety_preference", 84.5), "safety"
    )
    normalized_inputs["internet_speed_preference"] = encode_preference(
        user_input_dict.get("internet_speed_preference", 345.3), "internet_speed"
    )
    
    # Step 2: Apply transformation pipeline to all preference values
    # This will apply pre-fitted min-max scaling to the preference values
    # transform_pipeline is assumed to be imported from scaling_pipeline.py , but no script coded yet
    preferences_to_transform = {
         "climate_preference": normalized_inputs["climate_preference"],
         "cost_of_living_preference": normalized_inputs["cost_of_living_preference"],
         "healthcare_preference": normalized_inputs["healthcare_preference"],
         "safety_preference": normalized_inputs["safety_preference"],
         "internet_speed_preference": normalized_inputs["internet_speed_preference"]
    }
     
    # Import the pre-fitted pipeline and apply the transformation
    with open('../models/scaling_pipeline.pkl', 'rb') as f:
        pipe = pickle.load(f)
    
    transformed_preferences = pipe.transform(preferences_to_transform)
    
    # Update normalized inputs with transformed values
    for key, value in transformed_preferences.items():
        normalized_inputs[key] = value
    
    # Step 3: Scale importance values between 0 and 1
    importance_keys = [k for k in user_input_dict.keys() if k.endswith("_importance")]
    for key in importance_keys:
        if user_input_dict[key] is not None:
            normalized_inputs[key] = user_input_dict[key] / 10.0
        else:
            normalized_inputs[key] = 0.5  # Default importance if None
    
    # Step 4: Handle max_monthly_budget 
    # if user_input_dict.get("max_monthly_budget") is not None:
    #     # Apply same transformation as cost_of_living but remember pipeline
    #     # was not fitted on this column
    #     # Would need additional logic to properly scale this value
    #     # normalized_inputs["max_monthly_budget"] = transform_cost_of_living(
    #     #     user_input_dict["max_monthly_budget"]
    #     # )
    # else:
    #     # Default to a reasonable budget
    #     # normalized_inputs["max_monthly_budget"] = 0.5  # Mid-range scaled value
    
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
    if preference_type == "climate":
        if preference == "hot":
            return 25.0
        elif preference == "mild":
            return 18.0
        elif preference == "cold":
            return 11.0
        else:
            raise ValueError(f"Invalid climate preference: '{preference}'. Expected one of: 'hot', 'mild', 'cold'")

    elif preference_type == "cost_of_living":
        if preference == "low":
            return 400.0
        elif preference == "moderate":
            return 800.0
        elif preference == "high":
            return 2500.0
        else:
            raise ValueError(f"Invalid cost of living preference: '{preference}'. Expected one of: 'low', 'moderate', 'high'")

    elif preference_type == "healthcare":
        if preference == "excellent":
            return 75.0
        elif preference == "good":
            return 65.0
        elif preference == "fair":
            return 50.0
        else:
            raise ValueError(f"Invalid healthcare preference: '{preference}'. Expected one of: 'excellent', 'good', 'fair'")

    elif preference_type == "safety":
        if preference == "very_safe":
            return 75.0
        elif preference == "safe":
            return 65.0
        elif preference == "moderate":
            return 50.0
        else:
            raise ValueError(f"Invalid safety preference: '{preference}'. Expected one of: 'very_safe', 'safe', 'moderate'")

    elif preference_type == "internet_speed":
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




