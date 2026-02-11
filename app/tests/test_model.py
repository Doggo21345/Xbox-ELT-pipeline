import joblib
import pandas as pd
def test_model_acc():
    model = joblib.load("models/xbox_cluster_model.pkl")
    golden_data