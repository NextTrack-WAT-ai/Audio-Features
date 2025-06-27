import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from soundcloud_pipeline import SpotifyFeaturesTunable

# Load your CSV with Spotify features
df = pd.read_csv('music_info_cleaned.csv')

# List of files to process (ensure paths match your downloads)
file_paths = df['name'].astype(str) + ' - ' + df['artist'].astype(str) + '.mp3'
file_paths = file_paths.apply(lambda x: 'downloads/' + x)

# Initialize your analyzer/model
analyzer = SpotifyFeaturesTunable()

# Collect raw features and Spotify features
raw_features = []
spotify_danceability = []

for idx, row in df.iterrows():
    file_path = f"downloads/{row['artist']} - {row['name']}.mp3"
    try:
        base_feats = analyzer.precompute_base_features(file_path)
        # Example: use a subset of raw features for regression
        feats = [
            base_feats['beat_reg'],
            base_feats['bass_raw'],
            analyzer._normalize(base_feats['pulse_raw'], 0, 0.2, 0, 1)
        ]
        raw_features.append(feats)
        spotify_danceability.append(row['danceability'])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

X = np.array(raw_features)
y = np.array(spotify_danceability)

# Train regression model
reg = LinearRegression()
reg.fit(X, y)

print("Regression coefficients:", reg.coef_)
print("Regression intercept:", reg.intercept_)

# Predict on training data (or split for validation)
predicted = reg.predict(X)
print("First 10 predictions:", predicted[:10])
print("First 10 actual:", y[:10])