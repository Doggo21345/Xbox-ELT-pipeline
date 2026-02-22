import os
import pandas as pd
from pickle import dump
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import category_encoders as ce
import umap
import hdbscan
from sqlalchemy import create_engine, text
import toml

# Column definitions — mirrors EDA notebook
_COLS_TO_DROP = [
    'is_xpa', 'prices', 'Genre_x', 'Community_Notes',
    'lift_vs_rating_momentum', 'lift_vs_rating_discovery_capture',
    'zscore_discovery_capture', 'lift_vs_rating_quality_retention',
    'zscore_quality_retention', 'zscore_momentum', 'Removed', 'Metacritic',
    'Game', 'ProductID', 'url'
]
_LOW_CARD_COLS  = ['System', 'xCloud', 'Series_X_S', 'is_day_one_gp', 'party_type']
_HIGH_CARD_COLS = ['developer', 'publisher', 'Genre', 'ESRB_x', 'ESRB_Content_Descriptors']
_NUM_COLS = [
    'asset_count', 'rating_alltime_avg', 'rating_alltime_count',
    'current_price', 'days_since_release', 'days_since_gp_add',
    'momentum', 'discovery_capture', 'quality_retention'
]


def preprocess(df: pd.DataFrame):
    """
    Clean and encode the raw Hive DataFrame into a fully numeric matrix
    ready for UMAP. Returns (X_final, fitted_preprocessor).
    The preprocessor is saved alongside the models so the app can
    transform new data at inference time.
    """
    df = df.copy()

    # 1. Drop columns that add noise or are identifiers
    cols_to_drop = [c for c in _COLS_TO_DROP if c in df.columns]
    playcount_cols = [c for c in df.columns if 'PlayCount' in c]
    df = df.drop(columns=cols_to_drop + playcount_cols)
    print(f"   Dropped {len(cols_to_drop) + len(playcount_cols)} columns.", flush=True)

    # 2. Fill nulls
    if 'xCloud' in df.columns:
        df['xCloud'] = df['xCloud'].fillna('Not Supported')
    if 'developer' in df.columns and 'publisher' in df.columns:
        df['developer'] = df['developer'].fillna(df['publisher'])
    if 'publisher' in df.columns:
        df['publisher'] = df['publisher'].fillna('unknown')
    if 'developer' in df.columns:
        df['developer'] = df['developer'].fillna('unknown')

    # 3. Strip $ signs / non-numeric chars from price
    if 'current_price' in df.columns:
        df['current_price'] = (
            df['current_price']
            .astype(str)
            .str.replace(r'[^\d.]', '', regex=True)
        )

    # 4. Coerce numeric columns
    num_cols = [c for c in _NUM_COLS if c in df.columns]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. Build encoding pipeline
    low_card_cols  = [c for c in _LOW_CARD_COLS  if c in df.columns]
    high_card_cols = [c for c in _HIGH_CARD_COLS if c in df.columns]

    num_pipe = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler',  StandardScaler())
    ])
    low_card_pipe = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    # CountEncoder (frequency) handles high cardinality without needing a target
    high_card_pipe = Pipeline([
        ('freq_enc', ce.CountEncoder(normalize=True)),
        ('imputer',  KNNImputer(n_neighbors=5)),
        ('scaler',   StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num',       num_pipe,       num_cols),
        ('low_card',  low_card_pipe,  low_card_cols),
        ('high_card', high_card_pipe, high_card_cols),
    ], remainder='drop')

    X_processed  = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X_processed, columns=feature_names)

    # 6. Drop redundant OHE dummy (is_day_one_gp encodes as True/False pair)
    if 'low_card__is_day_one_gp_False' in X_df.columns:
        X_df = X_df.drop(columns=['low_card__is_day_one_gp_False'])

    # 7. Remove outliers with IsolationForest
    n_before = len(X_df)
    iso  = IsolationForest(contamination=0.01, random_state=42)
    mask = iso.fit_predict(X_df) == 1
    X_df = X_df[mask].reset_index(drop=True)
    print(f"   Outlier removal: {n_before - mask.sum()} rows removed, {len(X_df)} remaining.", flush=True)

    return X_df, preprocessor


def train_n_optimize(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir   = os.path.abspath(os.path.join(current_dir, "..", "models"))
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "xbox_model.pkl")

    # Preprocess raw data before training
    print("🔧 Preprocessing...", flush=True)
    X_final, preprocessor = preprocess(df)
    print(f"   Final feature matrix: {X_final.shape}", flush=True)

    # Grid search
    param_grid = {
        'n_neighbors':       [15, 30],
        'min_dist':          [0, .1],
        'min_cluster_size':  [20, 50],
        'min_samples':       [1, 5]
    }
    grid    = list(ParameterGrid(param_grid))
    results = []

    print(f"🚀 Starting optimisation on {len(X_final)} rows.")
    print(f"📊 Total combinations: {len(grid)}", flush=True)

    for i, params in enumerate(grid):
        print(f"🔄 [{i+1}/{len(grid)}] Testing: {params}...", flush=True)

        reducer = umap.UMAP(
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist'],
            n_components=10,
            random_state=42
        )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            gen_min_span_tree=True
        )

        embedding = reducer.fit_transform(X_final)
        labels    = clusterer.fit_predict(embedding)

        mask = labels != -1
        if mask.sum() > 1 and len(set(labels[mask])) > 1:
            sil = silhouette_score(embedding[mask], labels[mask])
        else:
            sil = -1

        results.append({'silhouette': sil, 'params': params, 'model': (reducer, clusterer)})
        print(f"   ✅ Score: {sil:.4f}", flush=True)

    # Save best result — includes preprocessor so inference works on new data
    best = max(results, key=lambda x: x['silhouette'])

    with open(save_path, 'wb') as f:
        dump({
            'preprocessor': preprocessor,
            'reducer':       best['model'][0],
            'clusterer':     best['model'][1],
            'score':         best['silhouette']
        }, f)

    print(f"\n✨ SUCCESS! Best Silhouette: {best['silhouette']:.4f}")
    print(f"💾 Saved to: {save_path}")
    return best['model']


if __name__ == "__main__":
    db_token          = os.getenv("DB_TOKEN")
    server_hostname   = os.getenv("DB_SERVER_HOSTNAME")
    http_path         = os.getenv("DB_HTTP_PATH")

    if not all([db_token, server_hostname, http_path]):
        try:
            secrets_path = os.path.join(os.path.dirname(__file__), "..", ".streamlit", "secrets.toml")
            secrets          = toml.load(secrets_path)
            db_token         = secrets["connections"]["databricks"]["access_token"]
            server_hostname  = secrets["connections"]["databricks"]["server_hostname"]
            http_path        = secrets["connections"]["databricks"]["http_path"]
            print("ℹ️  Using local secrets.toml")
        except Exception as e:
            print(f"❌ Error: No credentials found: {e}")
            raise

    connection_url = (
        f"databricks://token:{db_token}@{server_hostname}?"
        f"http_path={http_path}&catalog=hive_metastore&schema=default"
    )

    engine     = create_engine(connection_url)
    sql_string = "SELECT * FROM xbox_analysis_data"

    print(">>> STAGE 5: DOWNLOADING DATA FROM HIVE", flush=True)

    try:
        with engine.connect() as conn:
            result   = conn.execute(text(sql_string))
            final_df = pd.DataFrame(result.fetchall(), columns=result.keys())

        if final_df.empty:
            print("⚠️  The hive table returned no rows.")
        else:
            print(f"✅ Data downloaded: {len(final_df)} rows.", flush=True)
            train_n_optimize(final_df)

    except Exception as e:
        print(f"❌ Database Error: {e}")
        raise
