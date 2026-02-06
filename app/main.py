from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
from contextlib import asynccontextmanager
from model_utils import load_models, load_profiles, predict
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

app = FastAPI(title="Xbox Archetype API", lifespan=lifespan)

class GameFeatures(BaseModel):
    # --- Metadata & Categorical ---
    publisher: str = Field(..., example="electronic arts")
    developer: str = Field(..., example="respawn entertainment")
    Genre: str = Field(..., example="Action")
    party_type: str = Field(..., example="1st party")
    
    # --- Financials & Momentum ---
    current_price: float = Field(..., example=59.99)
    momentum: float = Field(..., example=0.85)
    days_since_release: int = Field(..., example=120)
    
    # --- Quality & Performance ---
    rating_alltime_avg: float = Field(..., example=4.2)
    rating_alltime_count: int = Field(..., example=1500)
    quality_retention: float = Field(..., example=0.75)
    
    # --- Xbox & Game Pass Specifics ---
    is_day_one_gp: int = Field(..., example=0, description="1 if Day One, 0 otherwise")
    days_since_gp_add: int = Field(..., example=45)
    xCloud: int = Field(..., example=1, description="Binary flag for cloud support")
    Series_X_S: int = Field(..., example=1)
    System: int = Field(..., example=1)
    
    # --- Engagement Metrics ---
    discovery_capture: float = Field(..., example=0.45)
    asset_count: int = Field(..., example=12)
    
    # --- Placeholder / Additional Features ---
    # Since you mentioned a 32-column preprocessor, 
    # ensure these match your Notebook's column list exactly.
    # If you have others (like 'sentiment' or 'dlc_count'), add them here:
    # sentiment_score: float = Field(0.0, example=0.5)

class RowInput(BaseModel):
    # This nests your new data type inside the 'features' key
    features: GameFeatures

class BatchInput(BaseModel):
    rows: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict") 
def predict_single(item: RowInput): 
    try:
        df = pd.DataFrame([item.features.model_dump()])
        preds = predict(df, state["preprocessor"], state["umap"], state["clusterer"])
        rec = preds.iloc[0].to_dict()
        label = int(rec["label"])
        if label == -1: 
            profile = {
                "name": "The Outlier / Unique Title",
                "description": "This game has a unique statistical DNA that doesn't fit a standard archetype.",
                "top_features": "Check individual metrics like momentum or price."
            }
        else: 
            profile = state["profiles"].get(str(label))
        return {
            "label": label, 
            "confidence": float(rec["confidence"]), 
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/batch_predict")
def batch_predict(item: BatchInput):
    try: 
        df = pd.DataFrame(item.features)
        results = []
        preds = predict(df, state["preprocessor"], state["umap"], state["clusterer"])
        for idx, row in preds.iterrows():
            profile = state["profiles"].get(str(label))
            rec = preds.iloc[0].to_dict()
            label = int(rec["label"])
            results.append({
                "profile": profile, 
                "label": label,
                "confidence": float(rec['confidence'])
            })
        return{"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail = str(e))