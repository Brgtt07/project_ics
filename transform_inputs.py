import numpy as np
from typing import Dict, Any, Optional

# This will be imported from the pipeline file
from scaling_pipeline import scale_pipeline

def transform_user_inputs(user_input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform user input dictionary into normalized values suitable for model prediction.
    
    Args:
        user_input_dict: Dictionary with user preferences and importance ratings
        
    Returns:
        Dictionary with normalized preference values and importance weights
    """
    normalized_inputs = {}
    
    # Step 1: Transform categorical preferences to numerical values
    # For climate preference
    if user_input_dict.get("climate_preference") is not None:
        climate_pref = user_input_dict["climate_preference"]
        if climate_pref == "hot":
            normalized_inputs["climate_preference"] = 27.0  # Higher average yearly temperature
        elif climate_pref == "moderate":
            normalized_inputs["climate_preference"] = 18.0  # Moderate temperature
        elif climate_pref == "cold":
            normalized_inputs["climate_preference"] = 10.0  # Low temperature
        else:
            normalized_inputs["climate_preference"] = 20.0  # Default value
    else:
        # Step 2: Default to good values when preference is None
        normalized_inputs["climate_preference"] = 20.0  # Default to pleasant temperature
    
    # For cost of living preference (lower is better)
    if user_input_dict.get("cost_of_living_preference") is not None:
        cost_pref = user_input_dict["cost_of_living_preference"]
        if cost_pref == "low":
            normalized_inputs["cost_of_living_preference"] = 400.0  # Lower monthly cost
        elif cost_pref == "moderate":
            normalized_inputs["cost_of_living_preference"] = 800.0  # Moderate cost
        elif cost_pref == "high":
            normalized_inputs["cost_of_living_preference"] = 1500.0  # Higher cost
        else:
            normalized_inputs["cost_of_living_preference"] = 600.0  # Default value
    else:
        normalized_inputs["cost_of_living_preference"] = 600.0  # Default to moderate cost
    
    # For healthcare preference (higher is better)
    if user_input_dict.get("healthcare_preference") is not None:
        healthcare_pref = user_input_dict["healthcare_preference"]
        if healthcare_pref == "excellent":
            normalized_inputs["healthcare_preference"] = 75.0  # High healthcare index
        elif healthcare_pref == "good":
            normalized_inputs["healthcare_preference"] = 65.0  # Good healthcare
        elif healthcare_pref == "fair":
            normalized_inputs["healthcare_preference"] = 50.0  # Fair healthcare
        else:
            normalized_inputs["healthcare_preference"] = 70.0  # Default value
    else:
        normalized_inputs["healthcare_preference"] = 70.0  # Default to good healthcare
    
    # For safety preference (higher is better)
    if user_input_dict.get("safety_preference") is not None:
        safety_pref = user_input_dict["safety_preference"]
        if safety_pref == "very_safe":
            normalized_inputs["safety_preference"] = 75.0  # High safety index
        elif safety_pref == "safe":
            normalized_inputs["safety_preference"] = 65.0  # Good safety
        elif safety_pref == "moderate":
            normalized_inputs["safety_preference"] = 50.0  # Moderate safety
        else:
            normalized_inputs["safety_preference"] = 70.0  # Default value
    else:
        normalized_inputs["safety_preference"] = 70.0  # Default to safe
    
    # For internet speed preference (higher is better)
    if user_input_dict.get("internet_speed_preference") is not None:
        internet_pref = user_input_dict["internet_speed_preference"]
        if internet_pref == "fast":
            normalized_inputs["internet_speed_preference"] = 200.0  # Fast internet
        elif internet_pref == "moderate":
            normalized_inputs["internet_speed_preference"] = 100.0  # Moderate speed
        elif internet_pref == "slow":
            normalized_inputs["internet_speed_preference"] = 50.0  # Slower internet
        else:
            normalized_inputs["internet_speed_preference"] = 150.0  # Default value
    else:
        normalized_inputs["internet_speed_preference"] = 300.0  # Default to very fast internet
    
    # Step 3: Apply transformation pipeline to all preference values
    # This will apply pre-fitted min-max scaling to the preference values
    # transform_pipeline is assumed to be imported from pipeline.py
    # preferences_to_transform = {
    #     "climate_preference": normalized_inputs["climate_preference"],
    #     "cost_of_living_preference": normalized_inputs["cost_of_living_preference"],
    #     "healthcare_preference": normalized_inputs["healthcare_preference"],
    #     "safety_preference": normalized_inputs["safety_preference"],
    #     "internet_speed_preference": normalized_inputs["internet_speed_preference"]
    # }
    # 
    # # Apply the transformation
    # transformed_preferences = transform_pipeline.transform(preferences_to_transform)
    # 
    # # Update normalized inputs with transformed values
    # for key, value in transformed_preferences.items():
    #     normalized_inputs[key] = value
    
    # Step 4: Scale importance values between 0 and 1
    importance_keys = [k for k in user_input_dict.keys() if k.endswith("_importance")]
    for key in importance_keys:
        if user_input_dict[key] is not None:
            normalized_inputs[key] = user_input_dict[key] / 10.0
        else:
            normalized_inputs[key] = 0.5  # Default importance if None
    
    # Step 5: Handle max_monthly_budget (commented as requested)
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


# Example usage
if __name__ == "__main__":
    example_user_inputs = {
        "climate_preference": "hot",
        "climate_importance": 10,
        "cost_of_living_preference": None,
        "cost_of_living_importance": 10,
        "max_monthly_budget": None,
        "healthcare_preference": None,
        "healthcare_importance": 5,
        "safety_preference": None,
        "safety_importance": 8,
        "internet_speed_preference": None,
        "internet_speed_importance": 8
    }
    
    normalized_inputs = transform_user_inputs(example_user_inputs)
    print(normalized_inputs) 