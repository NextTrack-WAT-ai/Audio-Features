import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify

from soundcloud_pipeline import SoundCloudPipeline
from run_pipeline import load_pipelines, vector_from_feats, bounded_targets

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

N_SONGS_TO_DOWNLOAD = 1000  # Adjust as needed
DOWNLOAD_FOLDER = Path("downloads")
DOWNLOAD_FOLDER.mkdir(exist_ok=True)  # Ensure download folder exists
# Initialize your analyzer once at startup
pipeline = SoundCloudPipeline(start_index=0, end_index=N_SONGS_TO_DOWNLOAD, download_folder=DOWNLOAD_FOLDER)
analyzer = pipeline.analyzer

# Load trained models once at startup from your model directory
MODEL_DIR = Path("models")
models = load_pipelines()

# Load track metadata for artist+track -> track_id lookup
METADATA_CSV = "music_info_cleaned.csv"
df = pd.read_csv(METADATA_CSV)
name_to_tid = {f"{r['artist']} - {r['name']}": r["track_id"] for _, r in df.iterrows()}
targets = {r["track_id"]: r.drop(["track_id", "artist", "name"]).to_dict() for _, r in df.iterrows()}

@app.route('/extract_features', methods=['POST'])
def extract_features():
    data = request.get_json()
    debug = request.args.get("debug") == "1"

    artist = data.get('artist')
    track_name = data.get('track_name')

    if not artist or not track_name:
        return jsonify({'error': 'Both "artist" and "track_name" must be provided'}), 400

    track_key = f"{artist} - {track_name}"
    logging.info(f"Received request for: {track_key}")

    # Find track_id from artist and track_name
    track_id = name_to_tid.get(track_key)
    if not track_id:
        msg = f'Track "{track_key}" not found in metadata.'
        logging.error(msg)
        return jsonify({'error': msg}), 404

    # Try to locate the audio file in the download folder based on track_key or track_id
    audio_file = None
    for file_path in pipeline.download_folder.glob("*.mp3"):
        if track_key.lower() in file_path.stem.lower():
            audio_file = file_path
            break
    if not audio_file:
        msg = f'Audio file for "{track_key}" not found in download folder.'
        logging.error(msg)
        return jsonify({'error': msg}), 404

    try:
        if debug:
            logging.info(f"Audio file path: {audio_file}")

        # Step 1: Extract base features
        base_feats = analyzer.precompute_base_features(str(audio_file))
        if base_feats is None:
            raise ValueError("Failed to extract base features")

        if debug:
            import pprint
            logging.info("=== Raw Extracted Features ===")
            pprint.pprint(base_feats)

        predictions = {}
        for target, pipe in models.items():
            X_raw = vector_from_feats(base_feats, target)

            if debug and target in {"acousticness", "instrumentalness"}:
                logging.info(f"\nVector for target '{target}':\n{X_raw}")

            if X_raw is None or np.isnan(X_raw).any() or X_raw.shape[1] == 0:
                predictions[target] = None
                continue

            pred = pipe.predict(X_raw)[0]
            if target == "key":
                pred = int(round(pred)) % 12
            elif target in bounded_targets:
                pred = float(np.clip(pred, 0, 1))
            else:
                pred = float(pred)

            predictions[target] = pred

        if debug:
            true_values = targets.get(track_id, {})
            logging.info("\n=== Prediction vs Ground Truth for '%s' ===", track_key)
            for tgt, pred_val in predictions.items():
                true_val = true_values.get(tgt)
                if true_val is None or pd.isna(true_val):
                    logging.info(f"{tgt:<15} Prediction: {pred_val:.4f}  |  Truth: N/A")
                else:
                    try:
                        diff = abs(pred_val - float(true_val))
                        logging.info(f"{tgt:<15} Prediction: {pred_val:.4f}  |  Truth: {true_val:.4f}  |  Δ = {diff:.4f}")
                    except Exception:
                        logging.info(f"{tgt:<15} Prediction: {pred_val}  |  Truth: {true_val} (non-numeric)")

        return jsonify(predictions)

    except Exception as e:
        logging.exception("Error processing extraction and prediction")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    logging.info("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True)
