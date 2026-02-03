from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from .model_utils import load_models, load_profiles, predict_batch_logic


app = FastAPI(title="Xbox Archtype API ")
class RowInput(BaseModel):
    features: Dict[str,Any]

class BatchInput(BaseModel):
    rows:List[Dict[str,Any]]
    
# 1. Global storage for models
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load everything once at startup
    preprocessor, umap, clusterer = load_models()
    profiles = load_profiles()
    models["preprocessor"] = preprocessor
    models["umap"] = umap
    models["clusterer"] = clusterer
    models["profiles"] = profiles
    print("🚀 Archetype Models Loaded and Ready")
    yield
    # Clean up (if needed)
    models.clear()

app = FastAPI(title="Xbox Archetype API", lifespan=lifespan)

# ... (Pydantic Models) ...

@app.get("/health")
def health():
    return{"status": "ok"}

@app.get("profiles")
def get_profiles():
    return load_profiles()

@app.post
def predict_single(item:RowInput)
    try:
        df = pd.DataFrame([item.features])
        preds = predict(df)
        rec = preds.iloc[0].to_dict()
        profiles = load_profiles()
        profile = profiles.get(str(int(rec["label"])), None) if rec["label"] != -1 else None
        return {"label": int(rec["label"]), "confidence": float(rec["confidence"]), "profile": profile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict_batch")
def predict_batch(batch: BatchInput):
    try:
        df = pd.DataFrame(batch.rows)
        preds = predict(df)
        results = []
        profiles = load_profiles()
        for idx, row in preds.iterrows():
            label = int(row['label'])
            results.append({
                "index": int(idx) if isinstance(idx, (int, np.integer)) else idx,
                "label": label,
                "confidence": float(row['confidence']),
                "profile": profiles.get(str(label)) if label != -1 else None
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
