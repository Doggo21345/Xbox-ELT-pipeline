import numpy as np
import pandas as pd
import pytest

from app.scripts.train import preprocess, train_n_optimize

# ── Fixture ──────────────────────────────────────────────────────────────────
# 100 rows: enough for KNNImputer (n_neighbors=5) and IsolationForest
# Deliberately includes messy data that mirrors what Hive actually sends:
#   - price as a "$xx.xx" string
#   - some NaN values in xCloud and developer
#   - columns that should be dropped (is_xpa, Game, PlayCount_*)

@pytest.fixture
def raw_df():
    rng = np.random.default_rng(42)
    n = 100

    prices = [
        f"${rng.uniform(0, 70):.2f}" if i % 4 != 0 else rng.uniform(0, 70)
        for i in range(n)
    ]
    xcloud_vals = np.where(rng.random(n) < 0.15, None,
                           rng.choice(["Supported", "Not Supported"], n))
    developer_vals = np.where(rng.random(n) < 0.1, None,
                              rng.choice(["Studio A", "Studio B", "Studio C"], n))

    return pd.DataFrame({
        # Numerics
        "asset_count":          rng.integers(1, 50, n).astype(float),
        "rating_alltime_avg":   rng.uniform(1, 5, n),
        "rating_alltime_count": rng.integers(10, 5000, n).astype(float),
        "current_price":        prices,
        "days_since_release":   rng.integers(30, 3000, n).astype(float),
        "days_since_gp_add":    rng.integers(1, 1000, n).astype(float),
        "momentum":             rng.uniform(-1, 1, n),
        "discovery_capture":    rng.uniform(-1, 1, n),
        "quality_retention":    rng.uniform(-1, 1, n),
        # Low-cardinality categoricals
        "System":               rng.choice(["Xbox One", "Xbox Series X|S", "PC"], n),
        "xCloud":               xcloud_vals,
        "Series_X_S":           rng.choice(["Optimized", "Compatible", "Not Available"], n),
        "is_day_one_gp":        rng.choice([True, False], n),
        "party_type":           rng.choice(["1st Party", "2nd Party", "3rd Party"], n),
        # High-cardinality categoricals
        "developer":            developer_vals,
        "publisher":            rng.choice(["Pub X", "Pub Y", "Pub Z"], n),
        "Genre":                rng.choice(["Action", "RPG", "Sports", "Strategy"], n),
        "ESRB_x":               rng.choice(["E", "T", "M", "E10+"], n),
        "ESRB_Content_Descriptors": rng.choice(["Violence", "Language", "None"], n),
        # Should be dropped
        "is_xpa":               rng.integers(0, 2, n),
        "Game":                 [f"Game {i}" for i in range(n)],
        "PlayCount_7days":      rng.integers(0, 1000, n),
        "PlayCount_30days":     rng.integers(0, 5000, n),
    })


# ── preprocess() tests ───────────────────────────────────────────────────────

def test_output_is_fully_numeric(raw_df):
    X, _ = preprocess(raw_df)
    non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    assert non_numeric == [], f"Non-numeric columns in output: {non_numeric}"


def test_price_dollar_signs_cleaned(raw_df):
    X, _ = preprocess(raw_df)
    price_cols = [c for c in X.columns if "current_price" in c]
    assert len(price_cols) > 0, "current_price column missing from output"
    assert X[price_cols[0]].dtype in [np.float32, np.float64]


def test_xcloud_nulls_filled(raw_df):
    raw_df.loc[:5, "xCloud"] = None
    # Should not raise — nulls are filled before encoding
    X, _ = preprocess(raw_df)
    assert not X.isnull().any().any()


def test_developer_falls_back_to_publisher(raw_df):
    raw_df.loc[0, "developer"] = None
    raw_df.loc[0, "publisher"] = "FallbackPub"
    X, _ = preprocess(raw_df)
    assert not X.isnull().any().any()


def test_drops_noise_columns(raw_df):
    X, _ = preprocess(raw_df)
    for col in ["is_xpa", "Game", "is_xpa"]:
        assert col not in X.columns, f"Column '{col}' should have been dropped"


def test_drops_playcount_columns(raw_df):
    X, _ = preprocess(raw_df)
    playcount_cols = [c for c in X.columns if "PlayCount" in c]
    assert playcount_cols == [], f"PlayCount columns not dropped: {playcount_cols}"


def test_no_nulls_in_output(raw_df):
    X, _ = preprocess(raw_df)
    assert not X.isnull().any().any(), "Output contains NaN values"


def test_returns_fitted_preprocessor(raw_df):
    _, preprocessor = preprocess(raw_df)
    assert hasattr(preprocessor, "transform"), "Preprocessor must expose .transform() for inference"


def test_preprocessor_can_transform_new_rows(raw_df):
    # Critical for inference: the saved preprocessor must work on unseen rows.
    # At inference the API receives current_price as a float (Pydantic validates it),
    # and model_utils.py fills None categoricals with 'unknown' before calling transform.
    _, preprocessor = preprocess(raw_df)
    new_row = raw_df.iloc[:1].copy()
    new_row["current_price"] = 59.99      # float, as Pydantic always delivers
    new_row["xCloud"] = "unknown"         # model_utils fills None → 'unknown'
    result = preprocessor.transform(new_row)
    assert result.shape[0] == 1


def test_handles_missing_optional_columns(raw_df):
    # If some columns aren't in the Hive table, preprocess should still work
    df = raw_df.drop(columns=["ESRB_x", "ESRB_Content_Descriptors"])
    X, _ = preprocess(df)
    assert not X.empty


def test_outlier_removal_reduces_row_count(raw_df):
    X, _ = preprocess(raw_df)
    # IsolationForest(contamination=0.01) on 100 rows removes ~1
    assert len(X) < len(raw_df)


# ── train_n_optimize() input validation tests ─────────────────────────────────

def test_train_rejects_non_dataframe():
    with pytest.raises(TypeError):
        train_n_optimize([1, 2, 3])


def test_train_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        train_n_optimize(pd.DataFrame())
