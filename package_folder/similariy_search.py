"""
Module for finding similar countries using KNN-based similarity search.

This module provides functionality to find countries that best match user preferences
using a weighted K-nearest neighbors approach. It calculates similarity scores based
on Euclidean distance between scaled feature values and provides explanatory deltas
to explain why countries are selected.
"""

import os
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Any

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_similar_countries(processed_inputs: Dict[str, Any], n_neighbors: int = 10) -> pd.DataFrame:
    """
    Find countries that best match the user's preferences using KNN similarity.
    
    This function takes processed user inputs from transform_user_inputs() and finds
    the most similar countries based on weighted Euclidean distance. It returns a DataFrame
    with both similarity scores and deltas from the ideal values in original units.
    
    Args:
        processed_inputs: Dictionary of processed user preferences from transform_user_inputs()
        n_neighbors: Number of similar countries to return (default: 10)
    
    Returns:
        DataFrame with similar countries, including:
            - country: Country name
            - similarity_score: Overall similarity score (higher is better)
            - feature specific deltas in original units (e.g., cost_of_living_delta)
            - original feature values for each country
    """
    # Load the dataset
    data_path = os.path.join(project_dir, "raw_data", "merged_country_level", "scaled_merged_data_after_imputation.csv")
    data = pd.read_csv(data_path)
    
    # Load the original unscaled dataset for delta calculations
    original_data_path = os.path.join(project_dir, "raw_data", "merged_country_level", "merged_dataset_with_knn.csv")
    original_data = pd.read_csv(original_data_path)
    
    # Make sure both datasets have a common identifier column
    # Ensure the country column exists and is properly named in both datasets
    if 'Unnamed: 0' in data.columns and 'country' not in data.columns:
        data.rename(columns={'Unnamed: 0': 'country'}, inplace=True)
    if 'Unnamed: 0' in original_data.columns and 'country' not in original_data.columns:
        original_data.rename(columns={'Unnamed: 0': 'country'}, inplace=True)
    
    # Load scaling pipeline to get feature names and for any additional scaling needs
    pipeline_path = os.path.join(project_dir, 'models', 'scaling_pipeline.pkl')
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    # Extract feature names from the pipeline
    numeric_features = pipeline.named_steps['column_transformer'].transformers_[0][2]
    
    # Make a copy of the initial data to avoid modifying the original
    filtered_data = data.copy()
    
    # Filter based on max monthly budget if provided
    if "max_monthly_budget" in processed_inputs:
        max_budget_scaled = processed_inputs["max_monthly_budget"]
        # We need to filter on the scaled data since max_monthly_budget is scaled
        filtered_data = filtered_data[filtered_data['average_monthly_cost_$'] <= max_budget_scaled]
    
    # Filter by continent/country list if provided
    if processed_inputs.get("filtered_countries"):
        # Ensure we're comparing lowercase values for case-insensitive matching
        filtered_data = filtered_data[filtered_data["country"].str.lower().isin(
            [c.lower() for c in processed_inputs["filtered_countries"]])]
    
    # If we have no countries left after filtering, return an empty DataFrame
    if filtered_data.empty:
        # Create an empty DataFrame with the expected columns
        result_columns = ['country', 'similarity_score'] + [f"{f}_delta" for f in numeric_features] + [f"{f}_original" for f in numeric_features]
        return pd.DataFrame(columns=result_columns)
    
    # Create a dictionary with the user's preferences for numeric features
    ideal_values = {
        'average_monthly_cost_$': processed_inputs["cost_of_living_preference"],
        'average_yearly_temperature': processed_inputs["climate_preference"],
        'internet_speed_mbps': processed_inputs["internet_speed_preference"],
        'safety_index': processed_inputs["safety_preference"],
        'Healthcare Index': processed_inputs["healthcare_preference"]
    }
    
    # Create weights array from importance ratings
    weights = np.ones(len(numeric_features))
    for i, feature in enumerate(numeric_features):
        # Map feature to its corresponding importance
        if feature == 'average_monthly_cost_$':
            importance_key = 'cost_of_living_importance'
        elif feature == 'average_yearly_temperature':
            importance_key = 'climate_importance'
        elif feature == 'internet_speed_mbps':
            importance_key = 'internet_speed_importance'
        elif feature == 'safety_index':
            importance_key = 'safety_importance'
        elif feature == 'Healthcare Index':
            importance_key = 'healthcare_importance'
        else:
            continue
        
        # Use the importance value if available, otherwise keep default of 1.0
        if importance_key in processed_inputs:
            weights[i] = processed_inputs[importance_key]
    
    # Create a vector of the ideal values in same order as features
    ideal_vector = np.array([ideal_values[feature] for feature in numeric_features])
    
    # Extract feature data as numpy array from filtered data
    feature_data = filtered_data[numeric_features].values
    
    # Calculate weighted Euclidean distances
    distances = np.sqrt(np.sum(weights * (feature_data - ideal_vector)**2, axis=1))
    
    # Convert distances to similarity scores (higher is better)
    max_distance = np.max(distances) if len(distances) > 0 else 1.0
    similarity_scores = 1 - (distances / max_distance)
    
    # Get the number of neighbors, but make sure we don't try to get more than we have
    n_neighbors = min(n_neighbors, len(filtered_data))
    
    # Get the top n_neighbors countries from the filtered data
    top_indices = np.argsort(distances)[:n_neighbors]
    
    # Create result DataFrame with countries, similarity scores from filtered data
    result = filtered_data.iloc[top_indices].copy()
    result['similarity_score'] = similarity_scores[top_indices]
    
    # Get the country names of the top results to match with original data
    top_countries = result['country'].tolist()
    
    # Get the corresponding rows from original_data based on country name
    # First, ensure all names are case-insensitive by making them lowercase
    original_data_lower = original_data.copy()
    original_data_lower['country_lower'] = original_data_lower['country'].str.lower()
    
    # Create a matching list of lowercase country names
    top_countries_lower = [c.lower() for c in top_countries]
    
    # Filter original data to include only countries in our result set
    original_matched_data = original_data_lower[original_data_lower['country_lower'].isin(top_countries_lower)]
    
    # Drop the temporary lowercase column
    original_matched_data = original_matched_data.drop('country_lower', axis=1)
    
    # Create a dictionary to map lowercase country names to their original data
    country_data_map = {}
    for _, row in original_matched_data.iterrows():
        country_data_map[row['country'].lower()] = row
    
    # Create a properly ordered DataFrame with exactly the same number of rows as result
    ordered_original_data = []
    for country in result['country']:
        country_lower = country.lower()
        if country_lower in country_data_map:
            ordered_original_data.append(country_data_map[country_lower])
    
    # Convert to DataFrame if we have matching data
    if ordered_original_data:
        original_matched_data = pd.DataFrame(ordered_original_data)
    else:
        # If no matches found, create an empty DataFrame with the same columns
        original_matched_data = pd.DataFrame(columns=original_data.columns)
    
    # Map back to original unscaled data for explanatory deltas
    # First, get the unscaled user preferences
    unscaled_preferences = {
        'average_monthly_cost_$': get_unscaled_value(processed_inputs["cost_of_living_preference"], 'average_monthly_cost_$', pipeline),
        'average_yearly_temperature': get_unscaled_value(processed_inputs["climate_preference"], 'average_yearly_temperature', pipeline),
        'internet_speed_mbps': get_unscaled_value(processed_inputs["internet_speed_preference"], 'internet_speed_mbps', pipeline),
        'safety_index': get_unscaled_value(processed_inputs["safety_preference"], 'safety_index', pipeline),
        'Healthcare Index': get_unscaled_value(processed_inputs["healthcare_preference"], 'Healthcare Index', pipeline)
    }
    
    # Add columns for deltas in original units
    for feature in numeric_features:
        if feature in original_matched_data.columns and len(original_matched_data) == len(result):
            original_feature_values = original_matched_data[feature].values
            ideal_value = unscaled_preferences[feature]
            delta_col_name = f"{feature}_delta"
            result[delta_col_name] = original_feature_values - ideal_value
            # Add the original feature values for reference
            result[f"{feature}_original"] = original_feature_values
            
            # Calculate individual feature match score (higher is better)
            # For each feature, calculate a normalized similarity score
            # Get the max delta across all countries for this feature for normalization
            all_deltas = np.abs(original_data[feature] - ideal_value)
            max_delta = np.max(all_deltas) if len(all_deltas) > 0 else 1.0
            
            # Avoid division by zero and normalize to 0-1 range
            if max_delta > 0:
                # Calculate match score (inverse of normalized absolute delta)
                # Smaller delta = higher score
                match_scores = 1 - (np.abs(result[delta_col_name]) / max_delta)
                # Ensure scores are within 0-1 range
                match_scores = np.clip(match_scores, 0, 1)
                # Store in result DataFrame
                result[f"{feature}_match_score"] = match_scores
            else:
                # If all countries have the same value (max_delta = 0), they all match perfectly
                result[f"{feature}_match_score"] = np.ones(len(result))
        else:
            # Handle case where feature doesn't exist or lengths don't match
            # Just set deltas to NaN to avoid errors
            result[f"{feature}_delta"] = np.nan
            result[f"{feature}_original"] = np.nan
            result[f"{feature}_match_score"] = np.nan
    
    # Return the sorted results (by similarity score)
    return result.sort_values('similarity_score', ascending=False)

def get_unscaled_value(scaled_value, feature_name, pipeline):
    """
    Convert a scaled feature value back to its original unscaled value.
    
    Args:
        scaled_value: The scaled value (0-1)
        feature_name: Name of the feature
        pipeline: The scaling pipeline used
    
    Returns:
        The unscaled value in original units
    """
    # Get the scaler from the pipeline
    column_transformer = pipeline.named_steps['column_transformer']
    transformers = column_transformer.transformers_
    
    # Find which transformer was used for this feature
    for name, transformer, features in transformers:
        # Convert features to list if needed
        if hasattr(features, 'tolist'):
            feature_list = features.tolist()
        else:
            feature_list = list(features)
            
        if feature_name in feature_list:
            # Get the index of the feature in this transformer
            feature_idx = feature_list.index(feature_name)
            
            # Get the min and max values used for scaling
            # In scikit-learn, MinMaxScaler uses data_min_ and data_max_
            min_val = transformer.data_min_[feature_idx]
            max_val = transformer.data_max_[feature_idx]
            
            # Perform inverse transform
            unscaled_value = scaled_value * (max_val - min_val) + min_val
            return unscaled_value
    
    # If we couldn't find the feature in any transformer, return the value as is
    return scaled_value