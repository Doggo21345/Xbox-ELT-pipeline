from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List,Dict,Any,Optional
import pandas as pd 
from .model_utils import load_models, load_profiles, predict 


app = FastAPI(title="Xbox Archtype API ")
class RowInput(BaseModel):
    features: Dict[str,Any]

class BatchInput(BaseModel):
    rows:List[Dict[str,Any]]

@app.get("/health")
def health():
    return{"status": "ok"}

@app.get("profiles")
def get_profiles():
    return load_profiles()

@app.post
def
