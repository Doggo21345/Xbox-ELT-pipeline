import joblib
import pandas as pd
from sklearn.metrics import silhouette_score
def test_model_acc():
    model = joblib.load("models/xbox_cluster_model.pkl")
    
    