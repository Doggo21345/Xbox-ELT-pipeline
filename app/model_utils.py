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

    # 1. FORCE TYPES: Convert numeric columns to float
    low_card = ['System', 'xCloud', 'Series_X_S', 'is_day_one_gp', 'party_type']
    high_card = ['developer', 'publisher', 'Genre', 'ESRB_x', 'ESRB_Content_Descriptors']
    num_cols = [
        'asset_count', 'rating_alltime_avg', 'rating_alltime_count',
        'current_price', 'days_since_release', 'days_since_gp_add',
        'momentum', 'discovery_capture', 'quality_retention'
    ]

    # 2. Force types to match the Blueprint
    for col in df.columns:
        if col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # IMPORTANT: TargetEncoder hates empty strings. 
            # We fill with 'unknown' which it saw during fit_transform (hopefully)
            df[col] = df[col].fillna('unknown').astype(str).str.strip().str.lower()

    # 3. Add placeholders for missing columns (like ESRB_x)
    for col in (low_card + high_card + num_cols):
        if col not in df.columns:
            df[col] = 0 if col in num_cols else 'unknown'

    # 4. Order columns for the ColumnTransformer
    df = df[num_cols + low_card + high_card]
    print(df.dtypes)
    try:
        X_enc = preprocessor.transform(df)
        emb = umap.transform(X_enc)
        labels, strengths = hdbscan.approximate_predict(clusterer, emb)
        
        return pd.DataFrame({
            "label": labels.astype(int),
            "confidence": strengths.astype(float)
        }, index=df.index)
    except Exception as e:
        print(f"❌ Transformation Error: {e}")
        raise e