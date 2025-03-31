from fastapi import FastAPI, Query, Body
from package_folder.scaling_pipeline import transform_user_inputs  # Make sure these are implemented
from package_folder.weighted_sum import weighted_sum  # Make sure these are implemented
import pandas as pd
app = FastAPI()

@app.get('/')
def root():
    return {'hello': 'world'}



@app.get("/recommend-countries")
def recommend_countries(
    user_inputs: dict = Query(..., description="User input data")
    ):
    
    # Step 1: Convert user inputs to numerical weights
    processed_inputs = transform_user_inputs(user_inputs)

    # Step 2: Score countries based on user preferences
    data = pd.read_csv("../raw_data/merged_country_level/scaled_merged_data_after_imputation.csv")
    result_df = weighted_sum(data, processed_inputs)

    # Step 3: Return top 5 countries (currently the weighted sum is already returning the top 5 countries sorted)
    #top_5 = result_df.sort_values(by="country_user_score", ascending=False).head(5)

    return result_df.to_dict(orient="records")











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
