import joblib
import pandas as pd
import numpy as np
import hdbscan
import os
import json

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

def load_models(models_dir=MODELS_DIR):
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))
    umap = joblib.load(os.path.join(models_dir, "umap_model.pkl"))
    clusterer = joblib.load(os.path.join(models_dir, "hdbscan_model.pkl"))
    return preprocessor, umap, clusterer

def load_profiles(artifacts_dir=ARTIFACTS_DIR):
    with open(os.path.join(artifacts_dir, "archetype_profiles.json"), "r") as f:
        return json.load(f)

def predict(df, preprocessor=None, umap=None, clusterer=None):
    """
    df: pandas DataFrame of raw features (same columns expected by preprocessor)
    returns: DataFrame with columns ['label','confidence'] appended
    """
    if preprocessor is None or umap is None or clusterer is None:
        preprocessor, umap, clusterer = load_models()
    X_enc = preprocessor.transform(df)
    # if preprocessor returns numpy array, re-create DataFrame indexes
    emb = umap.transform(X_enc)  # use transform for production
    # approximate_predict is recommended for new points
    labels, strengths = hdbscan.approximate_predict(clusterer, emb)
    out = pd.DataFrame({
        "label": labels.astype(int),
        "confidence": strengths.astype(float)
    }, index=df.index)
    return out