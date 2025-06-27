# tune_all_targets.py
import subprocess
import sys

targets = [
    "danceability", "energy", "acousticness", "valence", "tempo",
    "loudness", "instrumentalness", "speechiness", "liveness", "key"
]

for target in targets:
    print(f"Tuning: {target}")
    subprocess.run([sys.executable, "tune_model.py", target], check=True)
