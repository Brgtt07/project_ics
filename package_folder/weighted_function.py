import pandas as pd

df = pd.read_csv("../raw_data/merged_country_level/scaled_merged_data.csv")

def rank_countries(data: df, user_weights: dict) -> pd.DataFrame:
    # put user's input weights between 1-10
    normalized_weights = {feature: weight / 10 for feature, weight in user_weights.items()}

    # choosing those input features that are in the dataset
    relevant_features = [feature for feature in normalized_weights.keys() if feature in data.columns]

    # return empty dataFrame if no feature matches
    if not relevant_features:
        return pd.DataFrame(columns=["country", "score"])

    # multiplying the selected features by the user weights and sum it
    data["score"] = data[relevant_features].mul(pd.Series(user_weights), axis=1).sum(axis=1)

    # sorting the countires in descending order of the calculated score
    ranked_data = data.sort_values(by="score", ascending=False).reset_index(drop=True)

    # return the top 5 coutries in dataFrame
    return ranked_data[["country", "score"]].head(5)
