import re
from pathlib import Path
import os
# Make Numba completely inert on CPUs without AVX-512 etc.
os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# ---- hard monkey-patch for safety ----
import types
import importlib
numba = importlib.import_module("numba")

def _noop_decorator(*args, **kwargs):
    def wrapper(fn):
        return fn
    return wrapper

for _name in ("jit", "njit", "vectorize", "guvectorize", "stencil", "generated_jit"):
    setattr(numba, _name, _noop_decorator)

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import median_abs_deviation
import joblib
import logging

# --- SoundCloud-specific utilities -----------------------------------------
from soundcloud_pipeline import (
    SoundCloudPipeline,
    ClippedRegressor,
    LogTransformedRegressor,
    CyclicKeyRegressor,
)
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SONGS = 300
TUNING_CSV = "tuning_data.csv"
SEED = 42
TEST_TRACK = "downloads/Johnny Cash - Hurt.mp3"  # example inference track
MUSIC_INFO_CSV = "music_info_cleaned.csv"        # ground-truth Spotify data
SCALER_DIR = "scalers"
MODEL_DIR = "models"

os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def parse_float(val):
    if isinstance(val, (float, int, np.floating, np.integer)):
        return float(val)
    if isinstance(val, str):
        m = re.match(r"\[?([\d\.eE+\-]+)\]?", val.strip())
        if m:
            return float(m.group(1))
        try:
            return float(val)
        except ValueError:
            return np.nan
    return np.nan

# ---------------------------------------------------------------------------
# Feature map: raw -> Spotify targets
# ---------------------------------------------------------------------------
feature_map = {
    "danceability": ["beat_reg", "bass_raw", "pulse_raw"],
    "energy": ["rms_mean", "entropy_raw", "dyn_range_raw"],
    "acousticness": ["harmonic_ratio_raw", "centroid_raw", "flatness_raw", "contrast_ratio_raw"],
    "valence": ["onset_env_mean", "centroid_raw", "rms_mean", "entropy_raw",
                "high_freq_raw", "decay_raw", "mfcc_mean_raw"],
    "tempo": ["tempo_raw", "rms_db_mean"],
    "loudness": ["rms_db_mean", "entropy_raw"],
    "instrumentalness": ["mfcc_var_raw", "pitch_var_raw"],
    "speechiness": ["zcr_raw", "mfcc_delta_var_raw"],
    "liveness": ["dyn_range_liveness_raw", "high_freq_raw", "decay_raw", "zcr_raw"],
    "key": ["key_profile"],
}

bounded_targets = {
    "danceability",
    "energy",
    "acousticness",
    "valence",
    "instrumentalness",
    "speechiness",
    "liveness",
}

alt_model_targets = {"valence", "instrumentalness", "liveness", "tempo", "loudness"}

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    force_train = True
    pipeline = SoundCloudPipeline(start_index=0, end_index=N_SONGS)

    # pipeline.download_songs()
    pipeline.save_tuning_csv(TUNING_CSV)
    df = pd.read_csv(TUNING_CSV)

    # === Valence presence & stats ===
    if "valence" in df.columns:
        n_valence_missing = df["valence"].isna().sum()
        n_valence_present = df["valence"].notna().sum()
        std_valence = df["valence"].std()
        logging.info(
            "[DBG] valence column: NaNs=%d  non-NaNs=%d  std=%.6f",
            n_valence_missing,
            n_valence_present,
            std_valence,
        )
    else:
        logging.warning("[DBG] 'valence' column not found in CSV!")

    # Additional valence features missing count (before dropping duplicates)
    if "valence" in feature_map:
        valence_feats = feature_map["valence"]
        if all(feat in df.columns for feat in valence_feats):
            missing_valence_feats_rows = df[valence_feats].isna().any(axis=1).sum()
            logging.info(
                "[DBG] valence features missing in %d rows out of %d",
                missing_valence_feats_rows,
                len(df),
            )
        else:
            missing_feats = [f for f in valence_feats if f not in df.columns]
            logging.warning("[DBG] valence feature columns missing: %s", missing_feats)

    if {"title", "artist"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["title", "artist"])
    else:
        df = df.drop_duplicates()

    logging.info("Loaded %d unique tracks for tuning", len(df))

    X_dict, y_dict = {}, {}
    for tgt, raw_feats in feature_map.items():
        X_rows, y_vals = [], []
        for i, row in df.iterrows():
            try:
                if tgt == "key":
                    features = [
                        parse_float(row[f"key_profile_{j}"]) for j in range(12)
                    ]
                else:
                    features = [parse_float(row.get(f, np.nan)) for f in raw_feats]

                label = parse_float(row.get(tgt, np.nan))

                # Valence-specific detailed debug logs
                if tgt == "valence":
                    if pd.isna(label):
                        logging.debug(
                            "[valence] Skipping row %d due to missing valence label",
                            i,
                        )
                        continue
                    if any(np.isnan(features)):
                        missing_feats = [
                            f
                            for f, v in zip(raw_feats, features)
                            if np.isnan(v)
                        ]
                        logging.debug(
                            "[valence] Skipping row %d due to missing features: %s",
                            i,
                            missing_feats,
                        )
                        continue
                else:
                    if any(np.isnan(features)) or pd.isna(label):
                        logging.debug(
                            "[SKIP] Row %d skipped for target '%s' — NaN in data",
                            i,
                            tgt,
                        )
                        continue

                X_rows.append(features)
                y_vals.append(label)
            except Exception as e:
                logging.debug("[ERROR] Row %d error for target '%s': %s", i, tgt, e)
                continue

        if not X_rows:
            logging.warning("No samples for target '%s' — skipping", tgt)
            continue

        X_arr = np.asarray(X_rows, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)

        # Debug summary for valence after data prep but before filtering
        if tgt == "valence":
            logging.info(
                "[valence] Prepared %d samples with std=%.6f",
                len(y_arr),
                np.std(y_arr),
            )

        var = np.std(y_arr)
        median_y = np.median(y_arr)
        mad = median_abs_deviation(y_arr, scale="normal")
        logging.info("Target variance (%s): %.4f", tgt, var)

        if var > 0:
            z = np.abs((y_arr - median_y) / (mad + 1e-8))
            mask = z <= 3
            X_arr = X_arr[mask]
            y_arr = y_arr[mask]

        if tgt == "valence":
            logging.info(
                "[valence] %d samples remain after outlier removal", len(y_arr)
            )

        if np.std(y_arr) < 1e-5 and not force_train:
            logging.warning("Target '%s' has near-constant values — skipping", tgt)
            continue

        X_dict[tgt] = X_arr
        y_dict[tgt] = y_arr

    # -----------------------------------------------------------------------
    # Model training
    # -----------------------------------------------------------------------
    models = {}
    for tgt, X in X_dict.items():
        y = y_dict[tgt]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )

        if tgt in alt_model_targets:
            base = HistGradientBoostingRegressor(max_iter=200, random_state=SEED)
        else:
            base = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, random_state=SEED
            )

        if tgt in bounded_targets:
            reg = make_pipeline(StandardScaler(), ClippedRegressor(base, 0.0, 1.0))
        elif tgt == "tempo":
            reg = make_pipeline(StandardScaler(), LogTransformedRegressor(base))
        elif tgt == "key":
            reg = make_pipeline(
                StandardScaler(), CyclicKeyRegressor(MultiOutputRegressor(base))
            )
        else:
            reg = make_pipeline(StandardScaler(), base)

        reg.fit(X_train, y_train)
        models[tgt] = reg

        # Save scaler and model pipeline
        scaler = reg.named_steps["standardscaler"]
        joblib.dump(scaler, os.path.join(SCALER_DIR, f"scaler_{tgt}.joblib"))
        joblib.dump(reg, os.path.join(MODEL_DIR, f"model_{tgt}.joblib"))

        preds = reg.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        logging.info(
            "Trained %-15s | Val RMSE: %.3f | Val R²: %.3f", tgt, rmse, r2
        )
        logging.info(
            "%s | Train μ=%.3f σ=%.3f | Pred μ=%.3f σ=%.3f",
            tgt,
            np.mean(y_train),
            np.std(y_train),
            np.mean(preds),
            np.std(preds),
        )

    # -----------------------------------------------------------------------
    # Demo inference
    # -----------------------------------------------------------------------
    logging.info("Running demo inference on '%s'", TEST_TRACK)
    base_feats = pipeline.analyzer.precompute_base_features(TEST_TRACK)

    def load_models_and_scalers():
        loaded_models, loaded_scalers = {}, {}
        for tgt in feature_map.keys():
            model_path = os.path.join(MODEL_DIR, f"model_{tgt}.joblib")
            scaler_path = os.path.join(SCALER_DIR, f"scaler_{tgt}.joblib")
            if os.path.isfile(model_path) and os.path.isfile(scaler_path):
                loaded_models[tgt] = joblib.load(model_path)
                loaded_scalers[tgt] = joblib.load(scaler_path)
        return loaded_models, loaded_scalers

    models, scalers = load_models_and_scalers()

    def build_test_vector(name, base_feats):
        if name == "key":
            X_raw = np.array(base_feats["key_profile"]).reshape(1, -1)
        else:
            vals = []
            for f in feature_map[name]:
                v = base_feats.get(f, 0)
                if isinstance(v, (list, np.ndarray)):
                    v = np.array(v).flatten()[0]
                vals.append(float(v))
            X_raw = np.array([vals], dtype=float)

        if name in scalers:
            return scalers[name].transform(X_raw)
        logging.warning("No scaler found for %s; using raw features.", name)
        return X_raw

    music_info = (
        pd.read_csv(MUSIC_INFO_CSV) if Path(MUSIC_INFO_CSV).is_file() else None
    )

    for tgt in feature_map:
        if tgt not in models:
            continue
        try:
            X_test = build_test_vector(tgt, base_feats)
            pred = models[tgt].predict(X_test)[0]

            if tgt in bounded_targets:
                pred = np.clip(pred, 0.0, 1.0)
            elif tgt == "key":
                pred = int(round(pred)) % 12

            truth = (
                music_info.loc[music_info["name"] == "Hurt", tgt].values[0]
                if music_info is not None
                and not music_info.loc[music_info["name"] == "Hurt", tgt].empty
                else "NA"
            )
            logging.info("[Pred] %-15s predicted=%s truth=%s", tgt, pred, truth)
        except Exception as e:
            logging.error("Inference error for %s: %s", tgt, e)

if __name__ == "__main__":
    main()
