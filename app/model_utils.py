import joblib
import pandas as pd
import numpy as np
import hdbscan
import os
import json
# BASE_DIR is good, let's use it
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point directly to folders inside 'app'
# Change this:
MODELS_DIR = os.path.join(BASE_DIR, "models")  # Removed the ".."
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts") # Removed the ".."

# Debug print so you can see exactly where it's looking in the terminal
print(f"📦 Model Path: {MODELS_DIR}")

def load_models(models_dir=MODELS_DIR):
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))
    umap = joblib.load(os.path.join(models_dir, "umap_model.pkl"))
    clusterer = joblib.load(os.path.join(models_dir, "hdbscan_model.pkl"))
    return preprocessor, umap, clusterer

def load_profiles(artifacts_dir=ARTIFACTS_DIR):
    with open(os.path.join(artifacts_dir, "archtype_profiles.json"), "r") as f:
        return json.load(f)

def predict(df, preprocessor=None, umap=None, clusterer=None):
    """
    df: pandas DataFrame of raw features
    returns: DataFrame with columns ['label','confidence']
    """
    if preprocessor is None or umap is None or clusterer is None:
        preprocessor, umap, clusterer = load_models()

    # 1. THE MISSING PIECE: Handle your EA Imputation logic here
    # This ensures consistency with those 2 NA values you found in training
    if 'publisher' in df.columns:
        df['publisher'] = df['publisher'].fillna('electronic arts')
    if 'developer' in df.columns:
        df['developer'] = df['developer'].fillna('electronic arts')

    # 2. TRANSFORM: Scale and Encode (Handles all 32 columns)
    X_enc = preprocessor.transform(df)
    
    # 3. PROJECT: Move to UMAP space
    emb = umap.transform(X_enc) 
    
    # 4. CLUSTER: Get labels and membership strengths
    # approximate_predict is the HDBSCAN standard for new data
    labels, strengths = hdbscan.approximate_predict(clusterer, emb)
    
    out = pd.DataFrame({
        "label": labels.astype(int),
        "confidence": strengths.astype(float)
    }, index=df.index)
    
    return out