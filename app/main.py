from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from contextlib import asynccontextmanager
from model_utils import load_models, load_profiles, predict
# 1. Global storage for models to keep them in RAM
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs ONCE when the server starts
    preprocessor, umap, clusterer = load_models()
    profiles = load_profiles()
    state["preprocessor"] = preprocessor
    state["umap"] = umap
    state["clusterer"] = clusterer
    state["profiles"] = profiles
    print("🚀 Xbox Archetype Models Loaded and Ready")
    yield
    state.clear()

# 2. Only ONE app definition with lifespan
app = FastAPI(title="Xbox Archetype API", lifespan=lifespan)

class RowInput(BaseModel):
    features: Dict[str, Any]

class BatchInput(BaseModel):
    rows: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict") # Path added
def predict_single(item: RowInput): # Colon added
    try:
        df = pd.DataFrame([item.features])
        # Pass the pre-loaded models from state
        preds = predict(df, state["preprocessor"], state["umap"], state["clusterer"])
        
        rec = preds.iloc[0].to_dict()
        label = int(rec["label"])
        
        # Get profile from pre-loaded state
        profile = state["profiles"].get(str(label)) if label != -1 else None
        
        return {
            "label": label, 
            "confidence": float(rec["confidence"]), 
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))