"""
Test script for the package_folder.similarity module.

This script simulates the data flow from the API endpoint:
1. Defines sample user inputs (preferences, importances, budget, continent).
2. Loads the preprocessing pipeline.
3. Transforms user inputs into scaled values using the input_processor.
4. Calls the find_similar_countries function.
5. Prints intermediate scaled values and the final results DataFrame.
"""

import pandas as pd
import numpy as np
import sys
import os
import traceback

# Import necessary functions
try:
    from package_folder.data_utils import load_pipeline, load_data
    from package_folder.input_processor import transform_user_inputs
    from package_folder.similarity import find_similar_countries
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print("Make sure you're running this from the project root or adjust the import paths.")
    sys.exit(1)

# --- 1. Define Sample User Inputs --- 
# Mimic the structure expected by the API endpoint
# Adjust these values to test different scenarios
sample_user_inputs = {
    # Preferences (typically slider values, e.g., 1-10 or similar, before scaling)
    "climate_preference": "mild" ,      # Moderate climate preference

    # Importances (typically slider values, e.g., 1-5 or 0-1)
    "cost_of_living_importance": 0.8,
    "safety_index_importance": 0.9,
    "internet_speed_mbps_importance": 0.6,
    "healthcare_index_importance": 0.7,
    "climate_importance": 0.5,

    # Optional budget and continent
    "max_monthly_budget": 1500, # Example budget in USD
    "continent_preference": "EU" # Example: Filter for Europe
}

print("--- Sample User Inputs ---")
print(sample_user_inputs)
print("\n" + "="*30 + "\n")

# --- 2. Load Pipeline --- 
pipeline = load_pipeline()
print("Pipeline loaded successfully.")

# Load data to check for duplicate countries
data_scaled, data_original = load_data()
print("\n--- Checking for Duplicate Countries ---")
country_counts = data_original['country'].value_counts()
duplicates = country_counts[country_counts > 1]
if len(duplicates) > 0:
    print(f"Warning: Found {len(duplicates)} duplicate country names:")
    print(duplicates)
else:
    print("No duplicate country names found.")

print("\n" + "="*30 + "\n")

# --- 3. Transform User Inputs --- 
scaled_preferences, scaled_weights, scaled_budget = transform_user_inputs(sample_user_inputs, pipeline)

print("--- Scaled Preferences (Ideal Vector - 0 to 1) ---")
print(scaled_preferences)
print("\n--- Scaled Weights (Importance - 0 to 1) ---")
print(scaled_weights)
print(f"\n--- Scaled Budget (0 to 1) ---")
print(scaled_budget)

print("\n" + "="*30 + "\n")

# --- 4. Find Similar Countries --- 
continent_code = sample_user_inputs.get("continent_preference")
n_neighbors = 5 # Number of recommendations desired

print(f"--- Finding Top {n_neighbors} Similar Countries ---")
print(f"Continent Filter: {continent_code}")

# Set pandas display options for better readability
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', 1000)

try:
    # Run the similarity function
    result_df = find_similar_countries(
        scaled_preferences=scaled_preferences,
        scaled_weights=scaled_weights,
        scaled_budget=scaled_budget,
        continent_code=continent_code,
        n_neighbors=n_neighbors
    )
    
    print("\n--- Results DataFrame ---")
    if not result_df.empty:
        # Display results
        print(result_df)
        
        # Example: Accessing specific details for the top country
        top_country = result_df.iloc[0]
        print(f"\n--- Details for Top Country ({top_country['country']}) ---")
        print(f"Overall Similarity Score: {top_country['similarity_score']:.4f}")
        for feature in scaled_preferences.keys():
            original_col = f"{feature}_original"
            delta_col = f"{feature}_delta"
            match_col = f"{feature}_match_score"
            if original_col in top_country and delta_col in top_country and match_col in top_country:
                print(f"  {feature}:")
                print(f"    Original Value: {top_country[original_col]}")
                print(f"    Delta from Ideal: {top_country[delta_col]:.2f}")
                print(f"    Feature Match Score: {top_country[match_col]:.4f}")
    else:
        print("No countries found matching the criteria.")
except Exception as e:
    print(f"Error in find_similar_countries: {e}")
    print("\nDetailed traceback:")
    traceback.print_exc()

print("\n" + "="*30 + "\n")
print("Test script finished.") 