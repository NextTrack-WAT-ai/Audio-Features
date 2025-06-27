import json
import optuna
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import median_abs_deviation
from run_pipeline import feature_map, prepare_train_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CACHE_FILE = Path("feature_extraction_cache.json")
META_FILE = Path("music_info_cleaned.csv")

# Load data
def load_data():
    raw_data = json.loads(CACHE_FILE.read_text())
    df = pd.read_csv(META_FILE)
    name_to_tid = {f"{r['artist']} - {r['name']}": r["track_id"] for _, r in df.iterrows()}
    targets = {r["track_id"]: r.drop(["track_id", "artist", "name"]).to_dict() for _, r in df.iterrows()}

    return raw_data, name_to_tid, targets

# Extract samples by target
def get_samples(raw_data, name_to_tid, targets, target):
    samples = []
    for path, feats in raw_data.items():
        stem = Path(path).stem
        tid = name_to_tid.get(stem)
        if tid and tid in targets and pd.notna(targets[tid].get(target)):
            samples.append({"features": feats, "target": targets[tid][target]})
    return samples

# Objective function for Optuna
def make_objective(X, y):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        model = GradientBoostingRegressor(**params, random_state=42)
        pipeline = make_pipeline(StandardScaler(), model)
        score = cross_val_score(pipeline, X, y, cv=5, scoring="neg_root_mean_squared_error").mean()
        return -score
    return objective

# Run the optimization
def tune_target(target):
    raw_data, name_to_tid, targets = load_data()
    samples = get_samples(raw_data, name_to_tid, targets, target)
    if not samples:
        logging.error(f"No samples found for target: {target}")
        return

    X, y = prepare_train_data(samples, target)
    if y.size == 0:
        logging.error(f"No valid training data for target: {target}")
        return

    logging.info(f"Running Optuna tuning for target: {target}")
    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(X, y), n_trials=50)

    logging.info(f"Best trial for {target}: {study.best_trial.value:.4f}")
    logging.info(f"Best params: {study.best_trial.params}")

    # Save best params
    out_file = Path("tuned_params.json")
    if out_file.exists():
        all_params = json.loads(out_file.read_text())
    else:
        all_params = {}
    all_params[target] = study.best_trial.params
    out_file.write_text(json.dumps(all_params, indent=2))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python tune_target.py <target_name>")
    else:
        tune_target(sys.argv[1])
