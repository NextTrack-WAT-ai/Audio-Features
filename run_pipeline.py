import logging
import pandas as pd
import json
from soundcloud_pipeline import SoundCloudPipeline, SpotifyFeaturesTunable, HyperparameterTuner

# --- Configuration ---
N_SONGS = 100
TUNING_CSV = "tuning_data.csv"
SEED = 42


def main():
    # 1. Initialize pipeline for first 100 songs
    pipeline = SoundCloudPipeline(start_index=0, end_index=N_SONGS)

    # 2. Download songs (skips already-downloaded due to checkpointing)
    pipeline.download_songs()

    # 3. Extract features and build DataFrame for tuning
    # pipeline.save_tuning_csv(TUNING_CSV)
    df = pd.read_csv('tuning_data.csv')
    df = df.drop_duplicates()  

    df.to_csv('tuning_data.csv', index=False)

    # 4. Load the DataFrame for tuning
    df = pd.read_csv(TUNING_CSV)

    # 5. Initialize model and tuner
    model = pipeline.analyzer  # Uses the same weights as in the pipeline
    tuner = HyperparameterTuner(model, df, val_frac=0.2, seed=SEED)

    # 6. Run hyperparameter tuning
    result = tuner.tune(maxiter=50, popsize=15)
    print(f"Training loss: {result.fun}")

    # 7. Validate on the held-out set
    val_loss = tuner.validate()
    print(f"Validation loss: {val_loss}")

    # 8. (Optional) Save tuned weights
    with open("tuned_weights.json", "w") as f:
        json.dump(model.weights, f, indent=2)
    print("Tuned weights saved to tuned_weights.json")


if __name__ == "__main__":
    main() 