from fastapi import FastAPI, Body
from package_folder.scaling_pipeline import transform_user_inputs
from package_folder.weighted_sum import weighted_sum
import pandas as pd
import os

base_path = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(base_path, "raw_data", "merged_country_level", "scaled_merged_data_after_imputation.csv")
df = pd.read_csv(csv_path)
app = FastAPI()

@app.get("/")
def root():
    return {"hello": "world"}


@app.post("/recommend-countries")
def recommend_countries(
    user_inputs: dict = Body(..., description="User input data")
):
    processed_inputs = transform_user_inputs(user_inputs)
    result_df = weighted_sum(df, processed_inputs)
    top_5 = result_df.sort_values(by="country_score", ascending=False).head(5)
    return top_5.to_dict(orient="records")


#test endpoint to use for debugging
@app.post('/test_user_input')
def test_user_input(
    user_input: dict = Body(..., description="User input data")
):
    # Return the received user input for testing purposes
    # This helps debug what data is being received from the frontend
    return {
        "received_data": user_input,
        "message": "Input received successfully"
    }
