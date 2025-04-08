import pandas as pd
import numpy as np
from typing import Dict, Optional, Any

# Import necessary components from data_utils
from package_folder.data_utils import load_data, load_pipeline, COUNTRY_CONTINENT_MAP

# --- Value Unscaling ---

def get_unscaled_value(scaled_value: float, feature_name: str, pipeline: Any) -> float:
    """Converts a scaled feature value back to its original unscaled value using the pipeline."""
    try:
        column_transformer = pipeline.named_steps['column_transformer']
        # Iterate through transformers to find the correct one for the feature
        for name, transformer, features in column_transformer.transformers_:
            # Handle potential SimpleImputer steps or other transformers without 'features' list
            if not hasattr(transformer, 'get_feature_names_out'): continue
            
            # Get the list of features handled by this transformer
            try:
                # Use get_feature_names_out if available (more robust)
                feature_list = list(transformer.get_feature_names_out(features))
            except AttributeError:
                 # Fallback for older versions or different transformer types
                 feature_list = list(features)

            if feature_name in feature_list:
                # Find the index of the feature *within this transformer's output*
                feature_idx = feature_list.index(feature_name)
                
                # Access scaler attributes (assuming MinMaxScaler)
                # Check if data_min_ and data_max_ exist
                if hasattr(transformer, 'data_min_') and hasattr(transformer, 'data_max_'):
                  min_val = transformer.data_min_[feature_idx]
                  max_val = transformer.data_max_[feature_idx]
                  # Avoid division by zero if max == min
                  if max_val == min_val:
                      return min_val # Return the constant value
                  return scaled_value * (max_val - min_val) + min_val
                else:
                     print(f"Warning: Transformer for {feature_name} doesn't have data_min_/data_max_.")
                     return scaled_value # Return scaled value if unscaling is not possible
                     
    except KeyError as e:
        print(f"Error accessing pipeline components during unscaling: {e}. Returning scaled value.")
    except Exception as e:
        print(f"Unexpected error during unscaling for {feature_name}: {e}. Returning scaled value.")
        
    # Fallback if feature not found or error occurred
    return scaled_value


# --- Similarity Calculation ---

def find_similar_countries(
    scaled_preferences: Dict[str, float],
    scaled_weights: Dict[str, float],
    scaled_budget: Optional[float] = None,
    continent_code: Optional[str] = None,
    n_neighbors: int = 10
) -> pd.DataFrame:
    """
    Finds similar countries based on scaled inputs, budget, and continent filters.

    Args:
        scaled_preferences: Dictionary of scaled preference values (0-1).
        scaled_weights: Dictionary of scaled importance weights (0-1).
        scaled_budget: Optional scaled maximum monthly budget (0-1).
        continent_code: Optional continent code (e.g., "EU", "AS").
        n_neighbors: Number of countries to return.

    Returns:
        DataFrame with top N countries, including similarity scores, original values,
        deltas, and individual feature match scores.
    """
    data_scaled, data_original = load_data()
    pipeline = load_pipeline() # Needed for unscaling

    # Ensure required columns exist
    numeric_features = list(scaled_preferences.keys())
    if not all(f in data_scaled.columns for f in numeric_features) or \
       not all(f in data_original.columns for f in numeric_features):
        missing_scaled = [f for f in numeric_features if f not in data_scaled.columns]
        missing_orig = [f for f in numeric_features if f not in data_original.columns]
        raise ValueError(f"Missing required feature columns. Scaled: {missing_scaled}, Original: {missing_orig}")

    # --- Filtering ---    
    filtered_data_scaled = data_scaled.copy()
    
    # 1. Filter by Continent
    if continent_code and isinstance(continent_code, str) and continent_code:
        countries_in_continent = [
            country for country, continent in COUNTRY_CONTINENT_MAP.items()
            if continent == continent_code.upper() # Ensure comparison is consistent case
        ]
        if countries_in_continent:
             # Ensure case-insensitive matching on the dataframe country names
            filtered_data_scaled = filtered_data_scaled[
                filtered_data_scaled["country"].str.lower().isin(countries_in_continent)
            ]
        else:
            print(f"Warning: No countries found for continent code: {continent_code}")
            # Decide behavior: return empty or proceed without continent filter?
            # Returning empty might be safer if continent is a hard requirement.
            # return _create_empty_result_df(numeric_features) 

    # 2. Filter by Budget (using scaled values)
    if scaled_budget is not None:
        budget_feature = 'average_monthly_cost_$'
        if budget_feature in filtered_data_scaled.columns:
            filtered_data_scaled = filtered_data_scaled[
                filtered_data_scaled[budget_feature] <= scaled_budget
            ]
        else:
            print(f"Warning: Budget feature '{budget_feature}' not found in scaled data. Cannot filter by budget.")

    if filtered_data_scaled.empty:
        print("No countries match the specified filters.")
        return _create_empty_result_df(numeric_features)

    # --- Similarity Calculation ---
    ideal_vector = np.array([scaled_preferences[feature] for feature in numeric_features])
    weights_array = np.array([scaled_weights[feature] for feature in numeric_features])
    feature_data = filtered_data_scaled[numeric_features].values

    distances = np.sqrt(np.sum(weights_array * (feature_data - ideal_vector)**2, axis=1))
    
    # Handle cases with zero distance (perfect match) or single result
    max_distance = np.max(distances) if len(distances) > 0 else 1.0
    if max_distance < 1e-9: # Avoid division by zero if all distances are effectively zero
         similarity_scores = np.ones_like(distances)
    else:
         similarity_scores = 1 - (distances / max_distance)

    # --- Prepare Results ---    
    n_neighbors = min(n_neighbors, len(filtered_data_scaled))
    # Argsort returns indices that would sort the array (ascending distance = descending similarity)
    top_indices = np.argsort(distances)[:n_neighbors]
    
    result_df = filtered_data_scaled.iloc[top_indices].copy()
    result_df['similarity_score'] = similarity_scores[top_indices]

    # --- Add Original Values, Deltas, and Match Scores ---
    result_df = _add_details_to_results(result_df, data_original, scaled_preferences, pipeline, numeric_features)

    return result_df.sort_values('similarity_score', ascending=False)

def _add_details_to_results(result_df: pd.DataFrame, data_original: pd.DataFrame, 
                           scaled_preferences: Dict[str, float], pipeline: Any, 
                           numeric_features: list) -> pd.DataFrame:
    """Helper to add original values, deltas, and match scores to the result DataFrame."""
    
    top_countries = result_df['country'].tolist()
    # Ensure case-insensitive matching when retrieving original data
    original_matched_data = data_original[
        data_original['country'].str.lower().isin([c.lower() for c in top_countries])
    ].set_index('country') # Index by country for potential lookup

    # Reindex to match the order and handle potential missing countries
    # Use the exact country names from result_df for reindexing
    original_matched_data = original_matched_data.reindex(top_countries)
    original_matched_data.reset_index(inplace=True) # Put country back as column

    unscaled_preferences = {
        feature: get_unscaled_value(scaled_preferences[feature], feature, pipeline)
        for feature in numeric_features
    }
    
    for feature in numeric_features:
        if feature in original_matched_data.columns:
            original_values = original_matched_data[feature].values
            ideal_value = unscaled_preferences[feature]
            
            # Ensure data types are numeric for calculations
            original_values_numeric = pd.to_numeric(original_values, errors='coerce')
            delta = original_values_numeric - ideal_value
            
            result_df[f"{feature}_original"] = original_values # Keep original potentially non-numeric
            result_df[f"{feature}_delta"] = delta

            # Calculate feature match score (0-1, higher is better)
            # Normalize based on the range of the *original* data for this feature
            feature_range = data_original[feature].max() - data_original[feature].min()
            feature_range = feature_range if pd.notna(feature_range) and feature_range > 1e-9 else 1.0
            
            abs_delta = np.abs(delta)
            # Use np.clip after division to ensure 0-1 range, handle NaN deltas
            match_score = 1 - np.clip(abs_delta / feature_range, 0, 1)
            result_df[f"{feature}_match_score"] = np.nan_to_num(match_score, nan=0.0) # Replace NaN score with 0
        else:
            # Handle case where feature is missing in original data
            result_df[f"{feature}_original"] = np.nan
            result_df[f"{feature}_delta"] = np.nan
            result_df[f"{feature}_match_score"] = 0.0 # Assign 0 score if data is missing
            
    # Select and order columns
    final_columns = ['country', 'similarity_score'] + \
                    [col for f in numeric_features for col in 
                     [f"{f}_original", f"{f}_delta", f"{f}_match_score"]]
    # Filter out any columns that might not exist if errors occurred
    final_columns = [col for col in final_columns if col in result_df.columns]
    return result_df[final_columns]

def _create_empty_result_df(numeric_features: list) -> pd.DataFrame:
    """Helper function to create an empty DataFrame with the expected columns."""
    cols = ['country', 'similarity_score'] + \
            [col for f in numeric_features for col in 
            [f"{f}_original", f"{f}_delta", f"{f}_match_score"]]
    return pd.DataFrame(columns=cols) 