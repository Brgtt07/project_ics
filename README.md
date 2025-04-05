# Ideal Country Selector

A data-driven application to help users find their ideal country to live in based on personal preferences like climate, cost of living, healthcare quality, safety, and internet speed.

## Project Overview

The Ideal Country Selector analyzes country data from multiple sources and recommends countries that best match user preferences. The application uses a weighted similarity algorithm that considers both user preferences and their importance ratings.

## Features

- **User Preference Input**: Select preferences for climate, cost of living, healthcare, safety, and internet speed
- **Importance Ratings**: Specify how important each factor is to you
- **Budget Filter**: Optionally set a maximum monthly budget
- **Personalized Recommendations**: Get a ranked list of countries that best match your preferences

## Data

The application uses merged data from various sources, which includes:
- Average monthly cost of living (USD)
- Average yearly temperature (°C)
- Internet speed (Mbps)
- Safety index (0-100)
- Healthcare index (0-100)

The data is processed and normalized to enable fair comparisons between different metrics.

## Project Structure

- `package_folder/`: Core application code
  - `scaling_pipeline.py`: Handles user input processing and scaling
  - `weighted_sum.py`: Implements the recommendation algorithm
  - `api_file.py`: FastAPI backend for the application
  - `frontend.py`: Streamlit frontend interface
- `raw_data/`: Contains the country datasets
- `models/`: Contains the scaling pipeline model
- `notebooks/`: Jupyter notebooks for data exploration and model development

## Technologies Used

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Data Processing**: pandas, scikit-learn
- **Deployment**: Docker, Google Cloud Run

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the backend API: `uvicorn package_folder.api_file:app --reload`
4. Run the frontend: `streamlit run package_folder/frontend.py`


