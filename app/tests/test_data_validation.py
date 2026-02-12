import pytest 
import numpy as np 
import pandas as pd 
import json 
import os
from datetime import datetime, timedelta

def calculate_day_one():
    release_date 
    

def test_raw_data_scheam():
    sample_path = 'tests?samples/xbox_api_sample.json'
    with open (sample_path, 'r') as f:
        raw_data = json.load(f)
        df = pd.json_normalize(raw_data)
        expected_cols = ['product_name', 'current_price', 'rating_7_days.RatingCount']
        for col in expected_cols:
            assert col in df.columns, f"ETL logic Error: {col} missing after normalization and therefore can not be included "

def test_day_one ():
    today_str = datetime.now().strftime('%Y-%m-%d')
    yesterday_str = (datetime.now()-timedelta(days =1)).strftime('%Y-%m-%d')
    test_df = pd.DataFrame({
        'product_name':['New Game', 'Old Game']
        'Added': [today_str,. yesterday_str]
    })
    result_df = calculate_day_one(test_df)
    assert result_df.loc[result_df['product_name'] == 'New Game', 'is_day_one_gp'].iloc[0] == True
    assert result_df.loc[result_df['product_name'] == 'Old Game', 'is_day_one_gp'].iloc[0] == False
