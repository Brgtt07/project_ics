"""
Verbose version of input_processor.py with detailed logging for debugging

This file copies the transform_user_inputs function from input_processor.py
but adds detailed logging to debug the budget scaling logic.
"""

import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('verbose_input_processor')

def transform_user_inputs_verbose(user_input_dict: Dict[str, Any], pipeline: Any) -> Tuple[Dict[str, float], Dict[str, float], Optional[float]]:
    """
    Verbose version of transform_user_inputs with detailed logging for debugging the budget scaling logic.
    
    Transforms raw user inputs into scaled values that can be used by the recommendation system.
    
    Args:
        user_input_dict: Dictionary from frontend with preferences and importances
        pipeline: The trained sklearn pipeline for scaling features
    
    Returns:
        Tuple of (scaled_preferences, scaled_weights, scaled_budget)
    """
    logger.info("Starting transform_user_inputs_verbose")
    logger.debug(f"Input user_input_dict: {user_input_dict}")
    
    # 1. Determine available features
    # Extract pipeline features
    try:
        pipeline_features = pipeline.feature_names_in_
        logger.debug(f"Using pipeline.feature_names_in_: {pipeline_features}")
    except AttributeError:
        try:
            # Alternative approach if feature_names_in_ is not available
            pipeline_features = pipeline.named_steps['column_transformer'].get_feature_names_out()
            logger.debug(f"Using column_transformer.get_feature_names_out(): {pipeline_features}")
        except Exception as e:
            # Last resort - hardcoded features if needed
            logger.error(f"Error getting feature names from pipeline: {e}")
            logger.warning("Using backup hardcoded feature list")
            pipeline_features = [
                'average_monthly_cost_$', 'climate_index', 'safety_index', 
                'healthcare_index', 'pollution_index', 'quality_of_life_index'
            ]
    
    # 2. Initialize dictionaries for storing scaled values
    scaled_preferences = {}
    scaled_weights = {}
    scaled_budget = None  # Default to None if not provided/valid

    # Create mapping of feature keys from frontend to corresponding features in the pipeline
    # Example: climate_preference -> climate_index
    feature_mapping = {
        'climate_preference': 'climate_index',
        'cost_of_living_preference': 'average_monthly_cost_$', 
        'safety_preference': 'safety_index',
        'healthcare_preference': 'healthcare_index',
        'pollution_preference': 'pollution_index',
        'quality_of_life_preference': 'quality_of_life_index'
    }
    
    # 3. Process each feature preference
    for key, feature_name in feature_mapping.items():
        # Skip features not in pipeline
        if feature_name not in pipeline_features:
            logger.warning(f"Feature {feature_name} not in pipeline features. Skipping.")
            continue

        # Use default value of 5 if preference not provided, ensure conversion to float
        preference = float(user_input_dict.get(key, 5.0))
        # Scale to 0-1 range (preferences are in 0-10 range)
        scaled_preferences[feature_name] = preference / 10.0
        logger.debug(f"Feature {feature_name} preference: {preference} -> scaled: {scaled_preferences[feature_name]}")

        # Get the importance key (e.g., climate_importance for climate_preference)
        importance_key = key.replace('preference', 'importance')
        
        # Use default importance of 5 if not provided, ensure conversion to float
        importance = float(user_input_dict.get(importance_key, 5.0))
        scaled_weights[feature_name] = max(0.0, min(1.0, importance / 10.0)) # Clamp between 0 and 1
        logger.debug(f"Feature {feature_name} importance: {importance} -> scaled: {scaled_weights[feature_name]}")

    # 4. Scale max_monthly_budget if provided
    logger.info("Starting budget scaling logic")
    budget_input = user_input_dict.get("max_monthly_budget")
    logger.debug(f"Raw budget input: {budget_input} (type: {type(budget_input)})")
    
    if budget_input is not None:
        try:
            budget = float(budget_input)
            logger.debug(f"Converted budget to float: {budget}")
            
            if budget > 0:
                logger.debug(f"Budget > 0, proceeding with scaling")
                # Create DataFrame with budget and dummy values for scaling
                # Ensure columns match the order expected by the transformer
                budget_df_dict = { 
                    feat: budget if feat == 'average_monthly_cost_$' else 0 
                    for feat in pipeline_features
                }
                logger.debug(f"Budget dataframe dictionary: {budget_df_dict}")
                
                budget_df = pd.DataFrame([budget_df_dict], columns=pipeline_features)
                logger.debug(f"Budget dataframe shape: {budget_df.shape}")
                logger.debug(f"Budget dataframe head:\n{budget_df.head().to_string()}")
                
                # Use the specific column transformer for consistency
                logger.debug("Accessing column_transformer from pipeline")
                column_transformer = pipeline.named_steps['column_transformer']
                
                logger.debug("Transforming budget dataframe")
                transformed_data = column_transformer.transform(budget_df)
                logger.debug(f"Transformed data shape: {transformed_data.shape}")
                logger.debug(f"Transformed data: {transformed_data}")
                
                # Find the index of the cost feature in the transformed data
                # Assuming average_monthly_cost_$ is the first feature transformed
                scaled_budget = transformed_data[0][0]
                logger.info(f"Final scaled budget: {scaled_budget}")
            else:
                logger.warning(f"Budget <= 0 ({budget}), setting scaled_budget to None")
                scaled_budget = None
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting budget value: {e}. Budget input: {budget_input}")
            logger.info("Setting scaled_budget to None due to error")
            scaled_budget = None
    else:
        logger.info("No budget provided, scaled_budget remains None")

    logger.info("Completed transform_user_inputs_verbose")
    return scaled_preferences, scaled_weights, scaled_budget 

# Add a test function to run just the budget scaling part
def test_budget_scaling(budget_value, pipeline):
    """Test only the budget scaling part of the function"""
    logger.info(f"Testing budget scaling with value: {budget_value}")
    
    # Minimal user input with just the budget
    user_input = {
        "max_monthly_budget": budget_value,
        "climate_preference": 5,
        "cost_of_living_preference": 5
    }
    
    # Run the full function and return just the budget part
    _, _, scaled_budget = transform_user_inputs_verbose(user_input, pipeline)
    logger.info(f"Budget scaling test result: {budget_value} -> {scaled_budget}")
    return scaled_budget

if __name__ == "__main__":
    # Run a simple test if file is executed directly
    from package_folder.data_utils import load_pipeline
    import sys
    
    # Try to get budget from command line argument
    budget = 2000
    if len(sys.argv) > 1:
        try:
            budget = float(sys.argv[1])
        except ValueError:
            budget = sys.argv[1]  # Keep as string for testing invalid inputs
    
    pipeline = load_pipeline()
    test_budget_scaling(budget, pipeline) 