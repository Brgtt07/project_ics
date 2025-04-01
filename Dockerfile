FROM python:3.10.6-slim

COPY models models
COPY package_folder package_folder
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY raw_data/merged_country_level/scaled_merged_data_after_imputation.csv

RUN pip install --upgrade pip
# RUN pip install -e . right now this is not needed since we don't have a setup.py file
RUN pip install -r requirements.txt

# Run container locally
# CMD uvicorn package_folder.api_file:app --reload --host 0.0.0.0

# Run container deployed -> GCP
CMD uvicorn package_folder.api_file:app --reload --host 0.0.0.0 --port $PORT
