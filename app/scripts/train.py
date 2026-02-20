import os 
import pandas as pd 
from pickle import dump
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import umap 
import hdbscan
from sqlalchemy import create_engine
import toml

def train_n_optimize(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Defining the paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(current_dir, "..", "app", "models"))
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "xbox_model.pkl")

    # 1. Define the grid FIRST
    param_grid = {
        'n_neighbors': [15, 30],
        'min_dist': [0, .1],
        'min_cluster_size': [20, 50],
        'min_samples': [1, 5]
    }
    grid = list(ParameterGrid(param_grid))
    results = []

    print(f"🚀 Starting Optimization on {len(df)} rows.")
    print(f"📊 Total combinations to test: {len(grid)}", flush=True)

    # 2. Single loop with enumeration for tracking
    for i, params in enumerate(grid):
        print(f"🔄 [{i+1}/{len(grid)}] Testing: {params}...", flush=True)
        
        reducer = umap.UMAP(
            n_neighbors=params['n_neighbors'], 
            min_dist=params['min_dist'],
            random_state=42 
        )
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            gen_min_span_tree=True
        )
        
        # This is the "Heavy" part that takes time
        embedding = reducer.fit_transform(df)
        labels = clusterer.fit_predict(embedding)
        
        mask = labels != -1 
        if mask.sum() > 1 and len(set(labels[mask])) > 1:
            sil = silhouette_score(embedding[mask], labels[mask])
        else:
            sil = -1
            
        results.append({
            'silhouette': sil, 
            'params': params, 
            'model': (reducer, clusterer)
        })
        print(f"   ✅ Score: {sil:.2f}", flush=True)

    # 3. Find and Save the best result
    best = max(results, key=lambda x: x['silhouette'])
    
    with open(save_path, 'wb') as f:
         dump({
            'reducer': best['model'][0],
            'clusterer': best['model'][1],
            'score': best['silhouette']
         }, f)
         
    print(f"\n✨ SUCCESS! Best Silhouette: {best['silhouette']:.2f}")
    print(f"💾 Saved to: {save_path}")
    return best['model']

# 2. Execution block at the VERY bottom
if __name__ == "__main__":
    db_token = os.getenv("DB_TOKEN")
    server_hostname = os.getenv("DB_SERVER_HOSTNAME")
    http_path = os.getenv("DB_HTTP_PATH")

    if not all([db_token, server_hostname, http_path]):
        try:
            # Fallback for local development
            secrets_path = os.path.join(os.path.dirname(__file__), "..", ".streamlit", "secrets.toml")
            secrets = toml.load(secrets_path)
            db_token = secrets["connections"]["databricks"]["access_token"]
            server_hostname = secrets["connections"]["databricks"]["server_hostname"]
            http_path = secrets["connections"]["databricks"]["http_path"]
            print("ℹ️ Using local secrets.toml")
        except Exception as e:
            print(f"❌ Error: No credentials found: {e}")
            raise

    connection_url = (
        f"databricks://token:{db_token}@{server_hostname}?"
        f"http_path={http_path}&catalog=hive_metastore&schema=default"
    )

    engine = create_engine(connection_url)
    query = "SELECT * FROM xbox_analysis_data"
    final_df = pd.read_sql(query, engine)

    # Now this call works because the function is defined above
    train_n_optimize(final_df)