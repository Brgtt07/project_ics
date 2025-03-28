from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI()

base_path = os.path.dirname(os.path.abspath(__file__))  # Pega o diretório atual onde o script está
csv_path = os.path.join(base_path, '..', 'raw_data', 'merged_country_level', 'scaled_merged_data.csv')

# Load preprocessed dataset
df = pd.read_csv(csv_path)

# Define a mapping for qualitative choices to numerical values
qualitative_mapping = {
    "Very Low": 0.1, "Low": 0.3, "Moderate": 0.5, "High": 0.7, "Very High": 0.9
}

class UserPreferences(BaseModel):
    weights: dict  # Importance given to each feature (1-10)
    choices: dict  # User's qualitative choices (e.g., "Safe", "Very Safe")
    num_countries: int  # Number of countries to return

@app.get('/')
def root():
    return {'message': 'API is running'}

#@app.get('/predict')
#def predict():
#    return {'hello': 'world'}

@app.post('/predict')
def rank_countries(user_prefs: UserPreferences):
    # Feature mapping between frontend choices and scaled csv column names
    feature_mapping = {
        "Safety": "safety_index",
        "Cost of Living": "average_monthly_cost_$",
        "Health Care Index": "Healthcare Index",
        "Internet Speed": "internet_speed_mbps",
        "Temperature": "average_yearly_temperature"
    }
    # Normalize weights (convert 1-10 scale to 0-1 scale)
    normalized_weights = {feature_mapping[feature]: weight / 10 for feature, weight in user_prefs.weights.items()}

    # Convert qualitative choices to numerical values
    numerical_choices = {feature_mapping[feature]: qualitative_mapping[choice] for feature, choice in user_prefs.choices.items()}

    # Ensure selected features exist in the dataset
    relevant_features = [feature for feature in normalized_weights.keys() if feature in df.columns]

    # Verify if there aren't valid features
    if not relevant_features:
        missing_features = [feature for feature in normalized_weights.keys() if feature not in df.columns]
        return {"error": f"No valid features selected. Missing: {', '.join(missing_features)}"}

    # Entry validation for 'num_countries'
    if user_prefs.num_countries < 1 or user_prefs.num_countries > len(df):
        return {"error": "Invalid number of countries requested"}

    # Compute scores using both weights and user choices
    df["score"] = df[relevant_features].mul(pd.Series(numerical_choices), axis=1).mul(pd.Series(normalized_weights), axis=1).sum(axis=1)

    # Sort countries by score
    ranked_df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

    # Return top N countries
    return ranked_df[["country", "score"]].head(user_prefs.num_countries).to_dict(orient="records")
