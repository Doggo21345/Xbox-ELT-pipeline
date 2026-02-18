import pytest 
import numpy as np 
import pandas as pd 
import json 
import os
from datetime import datetime, timedelta


def test_raw_data_scheam():
    sample_path = 'tests?samples/xbox_api_sample.json'
    with open (sample_path, 'r') as f:
        raw_data = json.load(f)
        df = pd.json_normalize(raw_data)
        expected_cols = ['product_name', 'current_price', 'rating_7_days.RatingCount']
        for col in expected_cols:
            assert col in df.columns, f"ETL logic Error: {col} missing after normalization and therefore can not be included "


