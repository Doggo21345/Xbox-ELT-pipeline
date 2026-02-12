from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
from contextlib import asynccontextmanager
from .model_utils import load_models, load_profiles, predict
state = {}

# ... inside your main execution ...
best_reducer, best_clusterer = train_and_optimize(df_from_hive)

# Save both steps of the pipeline so the UI can use them
with open('app/models/xbox_model.pkl', 'wb') as f:
    pickle.dump({'reducer': best_reducer, 'clusterer': best_clusterer}, f) 

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
    publisher: str
    developer: str
    Genre: str
    party_type: str
    current_price: float
    momentum: float
    days_since_release: int
    rating_alltime_avg: float
    rating_alltime_count: int
    quality_retention: float
    is_day_one_gp: int
    days_since_gp_add: int
    xCloud: int
    Series_X_S: int
    System: int
    discovery_capture: float
    asset_count: int
    ESRB_x: str
    ESRB_Content_Descriptors: str

    # This part sets the default values in Swagger UI
    model_config = {
        "json_schema_extra": {
            "example": {
                "publisher": "warner bros. games",
                "developer": "avalanche software",
                "Genre": "Action-Adventure",
                "party_type": "3rd Party",
                "current_price": 69.99,
                "momentum": 22.71,
                "days_since_release": 1088,
                "rating_alltime_avg": 4.5,
                "rating_alltime_count": 174600,
                "quality_retention": -0.3,
                "is_day_one_gp": 0,
                "days_since_gp_add": 1152,
                "xCloud": 1,
                "Series_X_S": 0,
                "System": 1,
                "discovery_capture": 0.44,
                "asset_count": 11,
                "ESRB_x": "T",
                "ESRB_Content_Descriptors": "UseAlc, FanVio, Blo, MilLan"
            }
        }
    }

class RowInput(BaseModel):
    # This nests your new data type inside the 'features' key
    features: GameFeatures

class BatchInput(BaseModel):
    games: List[GameFeatures]
    model_config = {
        "json_schema_extra": {
            "example": {
                "games": [
                    {
                        "publisher": "warner bros. games",
                        "developer": "avalanche software",
                        "Genre": "Action-Adventure",
                        "party_type": "3rd Party",
                        "current_price": 69.99,
                        "momentum": 22.71,
                        "days_since_release": 1088,
                        "rating_alltime_avg": 4.5,
                        "rating_alltime_count": 174600,
                        "quality_retention": -0.3,
                        "is_day_one_gp": 0,
                        "days_since_gp_add": 1152,
                        "xCloud": 1,
                        "Series_X_S": 0,
                        "System": 1,
                        "discovery_capture": 0.44,
                        "asset_count": 11,
                        "ESRB_x": "T",
                        "ESRB_Content_Descriptors": "UseAlc, FanVio, Blo, MilLan"
                    },
                    {
                        "publisher": "ubisoft",
                        "developer": "ubisoft montreal",
                        "Genre": "Action-Adventure",
                        "party_type": "3rd Party",
                        "current_price": 19.99,
                        "momentum": -3.19,
                        "days_since_release": 5930,
                        "rating_alltime_avg": 4.7,
                        "rating_alltime_count": 85000,
                        "quality_retention": 0.09,
                        "is_day_one_gp": 0,
                        "days_since_gp_add": 0,
                        "xCloud": 0,
                        "Series_X_S": 0,
                        "System": 0,
                        "discovery_capture": -0.005,
                        "asset_count": 15,
                        "ESRB_x": "M",
                        "ESRB_Content_Descriptors": "IntVio, Blo, SexCon, StrLan"
                    }
                ]
            }
        }
    }

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
def batch_predict(batch: BatchInput):
    try: 
        data = pd.DataFrame(g.model_dump() for g in batch.games)
        df = pd.DataFrame(data)
        results = []
        preds = predict(df, state["preprocessor"], state["umap"], state["clusterer"])
        for idx,row in preds.iterrows():
            confidence = float(row["confidence"])
            label = int(row["label"])
            profile = state["profiles"].get(str(label))
            results.append({
                "profile": profile, 
                "label": label,
                "confidence": confidence
            })
        return{"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail = str(e))