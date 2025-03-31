import pandas as pd

df = pd.read_csv("raw_data/merged_country_level/scaled_merged_data_after_imputation.csv")

def weighted_sum(data: pd.DataFrame, user_inputs: dict) -> pd.DataFrame:
    """

    Params:
    - data: Normalized dataset (scaled betweeen 0-1)
    - user_inputs: user preferences and their importance (between 0-1)

    Output:
    - DataFrame with "country_score" column
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
