# /Users/federico/code/Brgtt07/project_ics/notebooks/knn.ipynb
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def scale_data_and_get_pipeline(df, save_path=None):
    """
    Scales numeric features using MinMaxScaler and returns the scaled numeric data, 
    the pipeline, and the list of numeric feature names. Optionally saves the pipeline.
    """
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    
    # Pipeline to scale only numeric features
    preprocessor = ColumnTransformer(
        transformers=[('num', MinMaxScaler(), numeric_features)],
        remainder='drop'  # Only keep scaled numeric columns from this pipeline
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    # Fit the pipeline and transform the data
    scaled_numeric_data = pipeline.fit_transform(df)
    
    # Create DataFrame for the scaled numeric data
    df_numeric_scaled = pd.DataFrame(
        scaled_numeric_data,
        columns=numeric_features,
        index=df.index
    )
    
    # Save the pipeline if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(pipeline, f)
            
    return df_numeric_scaled, pipeline, numeric_features

def find_similar_countries(
    ideal_country_values, # Dictionary of ideal values {feature_name: value}
    df_numeric_scaled,    # DataFrame containing ONLY scaled numeric data
    original_df,          # Original DataFrame to retrieve non-numeric info
    numeric_features,     # List of numeric feature names (order matters)
    pipeline,             # Fitted pipeline used for scaling
    n_neighbors=10, 
    feature_weights=None  # List or array of weights
):
    """
    Finds similar countries using weighted Euclidean distance on scaled numeric features.
    """
    # Ensure ideal values dict has all numeric features
    if set(ideal_country_values.keys()) != set(numeric_features):
        raise ValueError("Keys in ideal_country_values must match numeric_features.")

    # Create DataFrame for the ideal country, ensuring correct column order
    ideal_country_df = pd.DataFrame([ideal_country_values])[numeric_features]

    # Scale the numeric features of the ideal country
    ideal_country_scaled = pipeline.transform(ideal_country_df).flatten()

    # Prepare weights array
    weights_array = np.array(feature_weights) if feature_weights is not None else np.ones(len(numeric_features))
    if len(weights_array) != len(numeric_features):
         raise ValueError(f"Weights length ({len(weights_array)}) must match numeric features length ({len(numeric_features)})")

    # Extract numeric data array from the main scaled dataframe
    df_numeric_array = df_numeric_scaled[numeric_features].values

    # Calculate weighted Euclidean distances: sqrt(sum(weights * (difference^2)))
    distances = np.sqrt(np.sum(weights_array * (df_numeric_array - ideal_country_scaled)**2, axis=1))

    # Get indices of nearest neighbors
    nearest_indices = np.argsort(distances)[:n_neighbors]
    
    # Get the neighbor data from the *original* dataframe and add the distance
    close_countries = original_df.iloc[nearest_indices].copy()
    close_countries['distance'] = distances[nearest_indices] 

    return close_countries

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    DATA_PATH = "../raw_data/merged_country_level/merged_dataset_with_knn.csv"
    PIPELINE_PATH = "../models/v2_scaler_pipeline.pkl" 
    N_NEIGHBORS = 100
    
    # Example Ideal Country Values (keys must match numeric features found later)
    ideal_country_values = {
        'average_monthly_cost_$': 720.0, 
        'average_yearly_temperature': 13.0,
        'internet_speed_mbps': 116.6, 
        'safety_index': 50.6, 
        'Healthcare Index': 75.7
    }

    # Weights (length must match numeric features found later, order matters)
    weights_input = [1.0, 1.0, 1.0, 1.0, 1.0] # Equal weight initially

    # --- Data Loading ---
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df_original = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df_original.shape[0]} countries, {df_original.shape[1]} features")

    # --- Scaling & Pipeline ---
    pipeline = None
    df_numeric_scaled = None
    numeric_features = None

    if os.path.exists(PIPELINE_PATH):
        print(f"Loading existing scaling pipeline from {PIPELINE_PATH}")
        with open(PIPELINE_PATH, 'rb') as f:
            pipeline = pickle.load(f)
        
        # Infer numeric features from the loaded pipeline
        try:
            numeric_features = pipeline.steps[0][1].transformers_[0][2] 
            print(f"Inferred numeric features from pipeline: {numeric_features}")
            # Apply the loaded pipeline to get scaled numeric data
            scaled_numeric = pipeline.transform(df_original) # Assumes pipeline has numeric features stored
            df_numeric_scaled = pd.DataFrame(scaled_numeric, columns=numeric_features, index=df_original.index)
        except Exception as e:
             print(f"Warning: Could not load numeric features or transform data from pipeline ({e}). Will rescale.")
             pipeline = None # Force rescaling

    if pipeline is None: # If loading failed or file didn't exist
        print(f"Creating new scaling pipeline and saving to {PIPELINE_PATH}...")
        df_numeric_scaled, pipeline, numeric_features = scale_data_and_get_pipeline(df_original, save_path=PIPELINE_PATH) 
        print(f"Using numeric features: {numeric_features}")
        print("Pipeline created and saved.")
        
    # --- Input Validation ---
    # Ensure the provided ideal values and weights match the numeric features derived from data/pipeline
    if set(ideal_country_values.keys()) != set(numeric_features):
         raise ValueError(f"Input ideal values keys {list(ideal_country_values.keys())} do not match data's numeric features {numeric_features}.")
    if len(weights_input) != len(numeric_features):
        raise ValueError(f"Input weights length ({len(weights_input)}) must match numeric features length ({len(numeric_features)})")

    # --- Find Neighbors ---
    print("\nFinding similar countries...")
    similar_countries = find_similar_countries(
        ideal_country_values=ideal_country_values, 
        df_numeric_scaled=df_numeric_scaled, 
        original_df=df_original, 
        numeric_features=numeric_features,
        pipeline=pipeline,
        n_neighbors=N_NEIGHBORS,
        feature_weights=weights_input 
    )

    # --- Display Results ---
    print(f"\nTop {N_NEIGHBORS} similar countries (based on weighted distance):")
    
    # Identify country name column (handle 'Unnamed: 0' or other variations)
    country_col_name = 'Country' # Default guess
    if 'Unnamed: 0' in df_original.columns and 'Country' not in df_original.columns:
         country_col_name = 'Unnamed: 0' 
    elif 'Country' not in df_original.columns:
         country_col_name = df_original.columns[0] # Fallback to first column
    
    # Select columns to display
    display_cols = [country_col_name, 'distance'] + numeric_features
    
    # Ensure display_cols exist in the result DataFrame
    display_cols = [col for col in display_cols if col in similar_countries.columns]

    print(similar_countries[display_cols].head(N_NEIGHBORS).to_string(index=False))