"""
Debug script to test the budget scaling functionality in input_processor.py

This script tests the budget scaling logic from lines 84-102 in input_processor.py
to verify it works correctly with different budget inputs.
"""
import pandas as pd
import sys
import os
import json
from pprint import pprint

# Add the parent directory to sys.path to allow importing package_folder modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from package_folder.input_processor import transform_user_inputs
from package_folder.data_utils import load_pipeline

def debug_budget_scaling():
    """Test the budget scaling functionality with various inputs"""
    
    # Load the pipeline
    print("Loading pipeline...")
    pipeline = load_pipeline()
    
    # Test cases for budget values
    test_budgets = [1000, 2000, 5000, 10000, None, "invalid", 0]
    
    print("\n==== Testing Budget Scaling ====\n")
    
    for budget in test_budgets:
        print(f"\nTesting budget: {budget}")
        
        # Create minimal user input with just the budget
        user_input = {
            "max_monthly_budget": budget,
            # Add some other preferences to make the input valid
            "climate_preference": 5,
            "cost_of_living_preference": 5
        }
        
        # Print user input
        print("\nUser input:")
        pprint(user_input)
        
        try:
            # Call the transform_user_inputs function
            scaled_preferences, scaled_weights, scaled_budget = transform_user_inputs(user_input, pipeline)
            
            # Print results
            print("\nResults:")
            print(f"  Scaled budget: {scaled_budget}")
            
            # Get the feature list from pipeline
            try:
                pipeline_features = pipeline.feature_names_in_
            except AttributeError:
                # Fallback if feature_names_in_ not available
                pipeline_features = pipeline.named_steps['column_transformer'].get_feature_names_out()
                
            # Create a dataframe with the same shape as the one used in the function
            if budget is not None and not isinstance(budget, str) and budget > 0:
                budget_df = pd.DataFrame([{ 
                    feat: budget if feat == 'average_monthly_cost_$' else 0 
                    for feat in pipeline_features
                }], columns=pipeline_features)
                
                print("\n  Debug information:")
                print(f"  Budget dataframe shape: {budget_df.shape}")
                print("  Budget dataframe preview: ")
                print(budget_df[['average_monthly_cost_$']].head())
                
                # Try to transform this dataframe directly
                try:
                    column_transformer = pipeline.named_steps['column_transformer']
                    transformed = column_transformer.transform(budget_df)
                    print(f"  Direct transformation result: {transformed[0][0]}")
                except Exception as e:
                    print(f"  Error in direct transformation: {e}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n==== Testing with real application flow ====\n")
    
    # Create a more complete user input similar to what the real application would use
    real_user_input = {
        "max_monthly_budget": 2000,
        "climate_preference": 7,
        "cost_of_living_preference": 8,
        "healthcare_preference": 9,
        "safety_preference": 6,
        "climate_importance": 9,
        "cost_of_living_importance": 10,
        "healthcare_importance": 7,
        "safety_importance": 8,
        "continent_preference": "EU"
    }
    
    print("Real user input:")
    pprint(real_user_input)
    
    try:
        # Transform the inputs
        scaled_preferences, scaled_weights, scaled_budget = transform_user_inputs(real_user_input, pipeline)
        
        print("\nResults:")
        print(f"Scaled budget: {scaled_budget}")
        print("Scaled preferences:")
        pprint(scaled_preferences)
        print("Scaled weights:")
        pprint(scaled_weights)
        
        # Integrate with similarity to see the budget filter in action
        from package_folder.similarity import find_similar_countries
        
        # Find similar countries (just like in the real application)
        result_df = find_similar_countries(
            scaled_preferences=scaled_preferences,
            scaled_weights=scaled_weights,
            scaled_budget=scaled_budget,
            continent_code=real_user_input.get("continent_preference"),
            n_neighbors=3  # Just get top 3 for debugging
        )
        
        # Check the results
        print("\nFiltered countries (with budget constraint):")
        if result_df.empty:
            print("No countries matched the criteria")
        else:
            print(f"Found {len(result_df)} countries:")
            # Print country name and score
            for _, row in result_df.iterrows():
                print(f"- {row['country']}: {row['similarity_score']:.4f}")
                if 'average_monthly_cost_$_original' in row:
                    print(f"  Original cost: {row['average_monthly_cost_$_original']}")
        
        # Compare with results without budget constraint
        print("\nResults without budget constraint:")
        no_budget_result = find_similar_countries(
            scaled_preferences=scaled_preferences,
            scaled_weights=scaled_weights,
            scaled_budget=None,  # No budget constraint
            continent_code=real_user_input.get("continent_preference"),
            n_neighbors=3
        )
        
        if no_budget_result.empty:
            print("No countries matched the criteria")
        else:
            print(f"Found {len(no_budget_result)} countries:")
            # Print country name and score
            for _, row in no_budget_result.iterrows():
                print(f"- {row['country']}: {row['similarity_score']:.4f}")
                if 'average_monthly_cost_$_original' in row:
                    print(f"  Original cost: {row['average_monthly_cost_$_original']}")
        
    except Exception as e:
        print(f"Error in real application flow: {e}")

if __name__ == "__main__":
    debug_budget_scaling() 