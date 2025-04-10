import os
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple, Any
import pycountry
import pycountry_convert as pc

# --- Constants ---
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH_SCALED = os.path.join(PROJECT_DIR, "raw_data", "merged_country_level", "scaled_merged_data_after_imputation.csv")
DATA_PATH_ORIGINAL = os.path.join(PROJECT_DIR, "raw_data", "merged_country_level", "final_merged_dataset_with_knn.csv")
PIPELINE_PATH = os.path.join(PROJECT_DIR, 'models', 'scaling_pipeline.pkl')

# --- Cached Data ---
_pipeline_cache = None
_data_cache = None

# --- Helper Functions ---

def get_country_continent_mapping() -> Dict[str, str]:
    """Creates a dictionary mapping lowercase country names to continent codes."""
    mapping = {}
    for country in pycountry.countries:
        try:
            mapping[country.name.lower()] = pc.country_alpha2_to_continent_code(country.alpha_2)
        except KeyError:
            # Handle countries pycountry_convert doesn't know (e.g., Kosovo)
            continue
    return mapping

COUNTRY_CONTINENT_MAP = get_country_continent_mapping()

def load_pipeline() -> Any:
    """Loads the pre-fitted scaling pipeline (with caching)."""
    global _pipeline_cache
    if _pipeline_cache is None:
        try:
            with open(PIPELINE_PATH, 'rb') as f:
                _pipeline_cache = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Pipeline file not found at {PIPELINE_PATH}")
            raise # Re-raise the error to signal failure
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise
    return _pipeline_cache

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the scaled and original datasets (with caching)."""
    global _data_cache
    if _data_cache is None:
        try:
            data_scaled = pd.read_csv(DATA_PATH_SCALED)
            data_original = pd.read_csv(DATA_PATH_ORIGINAL)

            # Standardize country column name
            for df in [data_scaled, data_original]:
                if 'Unnamed: 0' in df.columns and 'country' not in df.columns:
                    df.rename(columns={'Unnamed: 0': 'country'}, inplace=True)
                # Ensure country column is string type for reliable merging/lookup
                if 'country' in df.columns:
                    df['country'] = df['country'].astype(str)

            _data_cache = (data_scaled, data_original)
        except FileNotFoundError as e:
            print(f"Error: Data file not found. {e}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    return _data_cache
