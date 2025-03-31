
from fastapi import FastAPI, Query
import pandas as pd
from package_folder.api_file import process_user_input, weighted_sum  # Make sure these are implemented

app = FastAPI()

@app.get("/recommend-countries")
def recommend_countries(
    cost: float = Query(..., description="Cost preference"),
    climate: str = Query(..., description="Preferred climate"),
    healthcare: float = Query(..., description="Healthcare rating"),
    internet: float = Query(..., description="Internet speed preference")
):
    user_inputs = {
        "cost": cost,
        "climate": climate,
        "healthcare": healthcare,
        "internet": internet
    }

    # Step 1: Convert user inputs to numerical weights
    processed_inputs = process_user_input(user_inputs)

    # Step 2: Score countries based on user preferences
    result_df = weighted_sum(processed_inputs)

    # Step 3: Return top 5 countries
    top_5 = result_df.sort_values(by="country_user_score", ascending=False).head(5)

    return top_5.to_dict(orient="records")
