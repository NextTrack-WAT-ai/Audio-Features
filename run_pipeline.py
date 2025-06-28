import os
import importlib, inspect
numba = importlib.import_module("numba")

def _noop_decorator(*args, **kwargs):
    """
    Acts like every numba decorator but does *nothing*.
    Handles decorator–style, decorator–with-args, and direct-call forms.
    """
    # ── Direct call:  numba.jit(func, nopython=True) ────────────────────
    if len(args) and callable(args[0]) and all(
        not inspect.isclass(a) for a in args[1:]
    ):
        return args[0]

    # ── Decorator (with or without kwargs) ──────────────────────────────
    def real_decorator(fn):
        return fn
    return real_decorator

for _name in (
    "jit", "njit", "vectorize", "guvectorize",
    "stencil", "generated_jit"
):
    setattr(numba, _name, _noop_decorator)

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import median_abs_deviation
import joblib
import traceback

from soundcloud_pipeline import SoundCloudPipeline, CyclicKeyRegressor, LogTransformedRegressor, ClippedRegressor

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

EXPECTED_RAW_FEATURES = [
    "tempo_raw", "beat_reg", "bass_raw", "pulse_raw", "rms_mean",
    "entropy_raw", "dyn_range_raw", "harmonic_ratio_raw", "centroid_raw",
    "flatness_raw", "contrast_ratio_raw", "onset_env_mean", "rms_db_mean",
    "mfcc_var_raw", "pitch_var_raw", "zcr_raw", "mfcc_delta_var_raw",
    "dyn_range_liveness_raw", "high_freq_raw", "decay_raw"
]

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
FEATURE_CACHE_FILE = "feature_extraction_cache.json"
N_SONGS_TO_DOWNLOAD = 1000  # <-- set how many songs you want to download/extract
DOWNLOAD_FOLDER = Path("downloads")
TUNED_PARAMS_FILE = Path("tuned_params.json")

try:
    TUNED_PARAMS = json.loads(TUNED_PARAMS_FILE.read_text())
except FileNotFoundError:
    TUNED_PARAMS = {}

# Per-target feature subsets:
feature_map = {
    "danceability": ["beat_reg", "bass_raw", "pulse_raw"],
    "energy": ["rms_mean", "entropy_raw", "dyn_range_raw"],
    "acousticness": [
        'tempo_raw', 'beat_reg', 'bass_raw', 'pulse_raw',
        'rms_mean', 'entropy_raw', 'dyn_range_raw',
        'harmonic_ratio_raw', 'centroid_raw', 'flatness_raw',
        'contrast_ratio_raw', 'onset_env_mean', 'rms_db_mean',
        'mfcc_var_raw', 'pitch_var_raw', 'zcr_raw',
        'mfcc_delta_var_raw', 'dyn_range_liveness_raw',
        'high_freq_raw', 'decay_raw',
        # NEW features:
        'harmonic_to_percussive_ratio',
        'spectral_rolloff_50',
        'zcr_var',
        'spectral_bandwidth',
        'flatness_var',
        'mfcc_mean_1',
        'mfcc_mean_2',
        'mfcc_mean_3',
        'mid_band_energy',
        'onset_rate',
        "harmonic_to_noise_ratio",
        "spectral_flux_mean",
        "spectral_flux_var",
        "acoustic_ratio",
        "percussive_ratio",
        "rolloff_ratio",
        "spectral_entropy",
    ],
    # 'valence' features must exist in raw_data or else inference will warn & error
    "valence": ["onset_env_mean", "centroid_raw", "rms_mean", "entropy_raw", "high_freq_raw", "decay_raw", "mfcc_mean_1", "mfcc_mean_2", "mfcc_mean_3"],
    "tempo": [
        "tempo_raw", "rms_db_mean",
        "beat_reg", "pulse_raw"       # ← added
    ],
    "loudness": [
        "rms_db_mean", "entropy_raw",
        "rms_mean"                    # ← added
    ],
    "instrumentalness": [
        'tempo_raw', 'beat_reg', 'bass_raw', 'pulse_raw',
        'rms_mean', 'entropy_raw', 'dyn_range_raw',
        'harmonic_ratio_raw', 'centroid_raw', 'flatness_raw',
        'contrast_ratio_raw', 'onset_env_mean', 'rms_db_mean',
        'mfcc_var_raw', 'pitch_var_raw', 'zcr_raw',
        'mfcc_delta_var_raw', 'dyn_range_liveness_raw',
        'high_freq_raw', 'decay_raw',
        # NEW features:
        'harmonic_to_percussive_ratio',
        'spectral_rolloff_50',
        'zcr_var',
        'spectral_bandwidth',
        'flatness_var',
        'mfcc_mean_1',
        'mfcc_mean_2',
        'mfcc_mean_3',
        'mid_band_energy',
        'onset_rate',
    ],    
    "speechiness": [
        "zcr_raw", 
        "mfcc_delta_var_raw",
        "zcr_var",
        "mfcc_delta_mean",
        "spectral_bandwidth_var",
        "spectral_rolloff_90",
        "spectral_flatness_var",
    ],
    "liveness": ["dyn_range_liveness_raw", "high_freq_raw", "decay_raw", "zcr_raw"],
    "key": ["key_profile"],  # special handling below
}

bounded_targets = {
    "danceability", "energy", "acousticness", "valence",
    "instrumentalness", "speechiness", "liveness",
}

alt_model_targets = {
    "valence", "instrumentalness", "liveness", "tempo", "loudness"
}

def flatten_feature(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v)
        return np.nan if arr.size == 0 else float(np.mean(arr))
    return v

def prepare_train_data(samples, target):
    feats_to_use = feature_map.get(target, [])
    if len(feats_to_use) == 0 and target != "key":
        logging.warning(f"No features specified for target '{target}' in feature_map!")
        return np.empty((0, 0)), np.empty(0)  # early return with empty arrays

    X, y = [], []
    for s in samples:
        if target == "key":
            # === KEY TARGET SPECIAL CASE ===
            # Use all 12 dimensions of key_profile as separate features
            key_profile = s["features"].get("key_profile", None)
            if (
                key_profile is None or
                len(key_profile) != 12 or
                any(pd.isna(key_profile))
            ):
                continue
            row = list(key_profile)
        else:
            row = [flatten_feature(s["features"].get(f, np.nan)) for f in feats_to_use]
            if any(pd.isna(row)):
                continue
        X.append(row)
        y.append(s["target"])

    X = np.asarray(X)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()

    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]

    if y.size == 0:
        return X, y

    # MAD outlier removal
    med = np.median(y)
    mad = median_abs_deviation(y, scale='normal') + 1e-8
    keep = np.abs((y - med) / mad) <= 3
    return X[keep], y[keep]

def build_pipeline(target):
    # Try to get tuned params for this target, else None
    tuned = TUNED_PARAMS.get(target, None)

    if tuned:
        # Extract hyperparameters or fallback to defaults
        n_estimators = tuned.get("n_estimators", 250)
        max_depth = tuned.get("max_depth", 5)
        learning_rate = tuned.get("learning_rate", 0.05)
    else:
        if target in {"tempo", "loudness"}:
            n_estimators = 250
            max_depth = 5
            learning_rate = 0.05
        elif target in {"acousticness", "instrumentalness", "liveness"}:
            n_estimators = 400
            max_depth = 6
            learning_rate = 0.03
        elif target in alt_model_targets:
            # For HistGradientBoostingRegressor, tuned params could also be added but for simplicity:
            n_estimators = 200
            max_depth = None  # This regressor uses max_iter
            learning_rate = None
        else:
            n_estimators = 120
            max_depth = 4
            learning_rate = None

    if target in alt_model_targets:
        base = HistGradientBoostingRegressor(
            max_iter=n_estimators,
            random_state=42
        )
    else:
        # Build GradientBoostingRegressor only if learning_rate is set, else fallback
        if learning_rate is not None:
            base = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )
        else:
            base = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

    # Compose pipeline (same as before)
    if target == "key":
        model = CyclicKeyRegressor(MultiOutputRegressor(base))
        pipeline = make_pipeline(StandardScaler(), model)
    elif target in {"tempo", "acousticness", "instrumentalness"}:
        pipeline = make_pipeline(StandardScaler(), LogTransformedRegressor(base))
    elif target in bounded_targets:
        pipeline = make_pipeline(StandardScaler(), ClippedRegressor(base, 0.0, 1.0))
    else:
        pipeline = make_pipeline(StandardScaler(), base)

    return pipeline

def train_one_target(X, y, target):
    pipe = build_pipeline(target)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr.reshape(-1, 1) if target == "key" else y_tr)

    y_pred = pipe.predict(X_val)
    if target == "key":
        y_val = y_val.ravel()
        y_pred = y_pred.ravel()
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    r2 = r2_score(y_val, y_pred)

    logging.info(f"Validation Targets ({target}):    {np.array2string(y_val, precision=4)}")
    logging.info(f"Validation Predictions ({target}):{np.array2string(y_pred, precision=4)}")
    logging.info(f"Trained {target:<15} | Val RMSE: {rmse:.3f} | Val R²: {r2:.3f}")
    logging.info(f"[{target}] Predicted std dev = {np.std(y_pred):.6f}, True std dev = {np.std(y_val):.6f}")

    return pipe

def save_pipeline(pipe, target):
    joblib.dump(pipe, Path(MODEL_DIR) / f"model_{target}.joblib")

def load_pipelines():
    models = {}
    for target in feature_map.keys():
        p = Path(MODEL_DIR) / f"model_{target}.joblib"
        if p.is_file():
            models[target] = joblib.load(p)
            logging.info(f"Loaded pipeline for {target} from {p}")
    return models

def vector_from_feats(base_feats, target):
    feats_to_use = feature_map.get(target, [])
    if target == "key":
        # Ensure key_profile is exactly 12-dimensional
        key_profile = base_feats.get("key_profile", np.zeros(12))
        key_profile = np.array(key_profile)
        if key_profile.size != 12:
            if key_profile.size > 12:
                key_profile = key_profile[:12]
            else:
                key_profile = np.pad(key_profile, (0, 12 - key_profile.size), constant_values=0)
        return np.array([key_profile])
    else:
        vals = []
        nan_features = []
        for f in feats_to_use:
            v = base_feats.get(f, np.nan)
            if isinstance(v, (list, np.ndarray)):
                v = np.array(v).flatten()
                if v.size == 0:
                    v = np.nan
                else:
                    v = v[0]
            if pd.isna(v):
                nan_features.append(f)
            vals.append(float(v) if not pd.isna(v) else np.nan)
        if nan_features:
            logging.warning(f"Inference input for target '{target}' contains NaNs in features {nan_features}")
        return np.array([vals])

def convert_np_types(obj):
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def main():
    # Step 1: Initialize pipeline and download songs (if needed)
    pipeline = SoundCloudPipeline(start_index=0, end_index=N_SONGS_TO_DOWNLOAD, download_folder=DOWNLOAD_FOLDER)

    existing_files = list(DOWNLOAD_FOLDER.glob("*.mp3"))
    existing_count = len(existing_files)
    needed_count = N_SONGS_TO_DOWNLOAD - existing_count

    logging.info("Currently found %d audio files in %s", existing_count, DOWNLOAD_FOLDER)

    if needed_count <= 0:
        logging.info("Already have %d or more audio files — skipping download.", N_SONGS_TO_DOWNLOAD)
    else:
        logging.info("Downloading %d more tracks from SoundCloud to reach %d total …", needed_count, N_SONGS_TO_DOWNLOAD)
        # Adjust pipeline to fetch the next N songs
        pipeline = SoundCloudPipeline(start_index=existing_count, end_index=existing_count + needed_count, download_folder=DOWNLOAD_FOLDER)
        pipeline.download_songs()

    # Step 2: Extract features and save to cache
    CACHE_FILE = Path(FEATURE_CACHE_FILE)
    raw_data   = {}

    if CACHE_FILE.is_file():
        raw_data = json.loads(CACHE_FILE.read_text())

    updated = False
    for mp3_path in pipeline.download_folder.glob("*.mp3"):   # iterates over every .mp3
        mp3_str = str(mp3_path)
        if mp3_str in raw_data:          # already in cache – skip
            continue

        try:
            feats = pipeline.analyzer.precompute_base_features(mp3_str)
            raw_data[mp3_str] = feats
            updated = True
            logging.info("Extracted features for %s", mp3_path.name)
        except Exception as e:
            print(traceback.format_exc())

    # Persist the (possibly) updated cache
    if updated:
        cleaned_data = convert_np_types(raw_data)  # Convert numpy types to native Python types
        CACHE_FILE.write_text(json.dumps(cleaned_data, indent=2))
        logging.info("Feature cache updated: %s", CACHE_FILE)
    else:
        logging.info("Feature cache already up-to-date.")

    # Step 3: Load metadata CSV and prepare mapping
    df = pd.read_csv("music_info_cleaned.csv")
    name_to_tid = {f"{r['artist']} - {r['name']}": r["track_id"] for _, r in df.iterrows()}
    targets = {r["track_id"]: r.drop(["track_id", "artist", "name"]).to_dict() for _, r in df.iterrows()}

    # Step 4: Organize samples for training
    data_by_target = {t: [] for t in targets[next(iter(targets))].keys()}
    for path, feats in raw_data.items():
        # path here corresponds to track_id or filename? Make sure it's consistent:
        stem = Path(path).stem
        tid = name_to_tid.get(stem)
        if tid and tid in targets:
            for tgt, val in targets[tid].items():
                if pd.notna(val):
                    data_by_target[tgt].append({"features": feats, "target": val})

    # Step 5: Train models
    for tgt, samples in data_by_target.items():
        logging.info(f"Preparing training data for '{tgt}' — samples: {len(samples)}")
        X, y = prepare_train_data(samples, tgt)
        logging.info(f"Prepared X shape: {X.shape} , y shape: {y.shape} for target '{tgt}'")
        if y.size > 0:
            logging.info(f"[{tgt}] y stats — min={np.min(y):.6f}, max={np.max(y):.6f}, mean={np.mean(y):.6f}, std={np.std(y):.6f}")
        if X.size == 0 or len(X.shape) < 2 or X.shape[1] == 0:
            logging.warning(f"No training data for target '{tgt}', skipping.")
            continue

        pipe = train_one_target(X, y, tgt)
        save_pipeline(pipe, tgt)

    # Step 6: Load models and evaluate on *all* songs
    models = load_pipelines()

    results_by_target = {tgt: [] for tgt in models}

    for path, feats in raw_data.items():
        stem = Path(path).stem
        tid = name_to_tid.get(stem)
        if not tid or tid not in targets:
            continue
        true_vals = targets[tid]

        for tgt, pipe in models.items():
            try:
                X_raw = vector_from_feats(feats, tgt)
                if np.isnan(X_raw).any():
                    continue
                pred = pipe.predict(X_raw)[0]
                if tgt == "key":
                    pred = int(round(pred)) % 12
                elif tgt in bounded_targets:
                    pred = float(np.clip(pred, 0, 1))

                truth = true_vals.get(tgt)
                if truth is not None and not pd.isna(truth):
                    results_by_target[tgt].append((pred, truth))
            except Exception as e:
                logging.error(f"Error evaluating {tgt} on {path}: {e}")

    # Print error summaries
    print("\n=== Evaluation Summary ===")
    for tgt, results in results_by_target.items():
        if not results:
            continue
        preds, trues = zip(*results)
        preds, trues = np.array(preds), np.array(trues)
        mae = np.mean(np.abs(preds - trues))
        rmse = np.sqrt(np.mean((preds - trues) ** 2))
        r2 = r2_score(trues, preds)
        print(f"{tgt:<15} MAE={mae:.4f}  RMSE={rmse:.4f}  R^2={r2:.3f}")

if __name__ == "__main__":
    main()