"""
Module for calculating country recommendations based on user preferences.

This module implements the core recommendation algorithm for the Ideal Country Selector 
application. It uses a weighted sum approach to calculate similarity scores between user 
preferences and country data. The algorithm considers the importance weights provided by 
the user to prioritize certain factors over others.

Input data:
    - Normalized country dataset with metrics like cost of living, climate, healthcare, etc.
    - User preferences and importance weights for each metric
    
Output data:
    - Ranked list of countries with similarity scores
"""

import pandas as pd
import os

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def weighted_sum(data: pd.DataFrame, user_inputs: dict) -> pd.DataFrame:
    """
    Calculate country recommendations based on user preferences using a weighted similarity approach.
    
    This function computes similarity scores between user preferences and country data, 
    weighting each factor by its importance to the user. The algorithm calculates the 
    absolute difference between user preferences and country metrics, applies importance 
    weights, and normalizes the final scores. Countries are then ranked by their scores.
    
    Args:
        data: Normalized dataset (scaled between 0-1) containing country data with columns:
            - country: Country name
            - average_monthly_cost_$: Cost of living in USD (scaled)
            - average_yearly_temperature: Average temperature in °C (scaled)
            - internet_speed_mbps: Internet speed in Mbps (scaled)
            - safety_index: Safety index (scaled)
            - Healthcare Index: Healthcare quality index (scaled)
            
        user_inputs: Dictionary with user preferences and importance weights:
            - *_preference: User preference for each factor (scaled between 0-1)
            - *_importance: User-specified importance for each factor (scaled between 0-1)
            - max_monthly_budget: Optional maximum budget (scaled between 0-1)
    
    Returns:
        DataFrame with top 5 recommended countries, containing columns:
            - country: Country name
            - country_score: Similarity score (higher is better, max 1.0)
        The DataFrame is sorted by country_score in descending order.
    """

    feature_mapping = {
        "climate_preference": "average_yearly_temperature",
        "climate_importance": "average_yearly_temperature",
        "average_monthly_cost_$": "average_monthly_cost_$",
        "max_monthly_budget": "average_monthly_cost_$",
        "healthcare_preference": "Healthcare Index",
        "healthcare_importance": "Healthcare Index",
        "safety_preference": "safety_index",
        "safety_importance": "safety_index",
        "internet_speed_preference": "internet_speed_mbps",
        "internet_speed_importance": "internet_speed_mbps"
    }

    # interpret the user's preferences
    preferences = {}
    importance = {}

    for user_key, dataset_key in feature_mapping.items():
        if "preference" in user_key and user_key in user_inputs:
            preferences[dataset_key] = user_inputs[user_key]
        if "importance" in user_key and user_key in user_inputs:
            importance[dataset_key] = user_inputs[user_key]



    # check if the features are matching
    relevant_features = [feature for feature in preferences.keys() if feature in data.columns]

    if not relevant_features:
        return pd.DataFrame(columns=["country", "country_score"])  # if no matching return it empty

    # put the weights on
    scores = []
    for _, row in data.iterrows():
        total_score = 0
        total_weight = 0

        for feature in relevant_features:
            user_pref = preferences[feature]
            weight = importance.get(feature, 1)  # if no value, default is 1

            # count the differences (the absolute difference)
            difference = abs(row[feature] - user_pref)

            # adding the weights
            total_score += difference * weight
            total_weight += weight

        # Normalizing the score
        country_score = 1 - (total_score / total_weight if total_weight != 0 else 1)
        scores.append((row["country"], country_score))

    # return it in a dataframe
    ranked_data = pd.DataFrame(scores, columns=["country", "country_score"])
    ranked_data = ranked_data.sort_values(by="country_score", ascending=False).reset_index(drop=True)

    return ranked_data.head()
