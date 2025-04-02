from fastapi import FastAPI
from package_folder.scaling_pipeline import transform_user_inputs
from package_folder.weighted_sum import weighted_sum
import pandas as pd
import os

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()

@app.get('/')
def root():
    return {'hello': 'world'}

@app.post("/recommend-countries")
def recommend_countries(user_inputs: dict):
    # Process user inputs (encoding and normalization), returns a dictionary
    processed_inputs = transform_user_inputs(user_inputs)

    # Load the dataset
    data_path = os.path.join(project_dir, "raw_data", "merged_country_level", "scaled_merged_data_after_imputation.csv")
    data = pd.read_csv(data_path)

    #Filter the dataset based on the max_monthly_budget
    if 'max_monthly_budget' in processed_inputs:
        data = data[data['cost_of_living'] <= processed_inputs['max_monthly_budget']]

    # Calculate weighted scores
    result_df = weighted_sum(data, processed_inputs)

    return result_df.to_dict(orient="records")
