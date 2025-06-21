import os
import json
import time
import logging
import re
from pathlib import Path
from urllib.parse import quote, quote_plus
import pandas as pd
import requests
from bs4 import BeautifulSoup
from thefuzz import fuzz
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
import yt_dlp # Import yt-dlp
import librosa
import numpy as np
from scipy import stats
import warnings
from scipy.optimize import differential_evolution

# --- Configuration ---
DOWNLOAD_FOLDER = Path("downloads")
CHECKPOINT_FILE = DOWNLOAD_FOLDER / "song_url.json"
COMPARISONS_FOLDER = Path("comparisons")
SOUNDCLOUD_SEARCH_URL = "https://soundcloud.com/search?q={query}"
FEATURE_CACHE = "./audio_features_cache.csv"
SPOTIFY_BASELINE = "music_info_cleaned.csv"

REQUEST_TIMEOUT = 15  # seconds for HTTP requests
SELENIUM_TIMEOUT = 30 # Increased timeout seconds for Selenium waits
MATCH_THRESHOLD = 75  # Minimum fuzzy match score (0-100)
REQUEST_DELAY = 2     # seconds delay between requests

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def sanitize_filename(filename):
    """Removes characters invalid for filenames."""
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace sequences of whitespace with a single space
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    # Limit length if necessary (optional)
    # max_len = 200 
    # sanitized = sanitized[:max_len]
    return sanitized

def normalize(s):
    return re.sub(r'\W+', '', s).strip().lower()

# def compare_results(your_feats, spotify_feats, song, artist):
#     features = [
#         'danceability', 'energy', 'acousticness', 'instrumentalness',
#         'liveness', 'valence', 'speechiness', 'tempo', 'loudness', 'key'
#     ]

#     your_vals = [your_feats[f] for f in features]
#     spotify_vals = [spotify_feats[f] for f in features]

#     comp_df = pd.DataFrame({
#         'Feature': features,
#         'Your Algorithm': your_vals,
#         'Spotify': spotify_vals
#     })
#     comp_df['Difference'] = comp_df['Your Algorithm'] - comp_df['Spotify']

#     # key error score row
#     predicted_key = feats['key']
#     spotify_key = int(obs['key'])
#     key_dist = abs(predicted_key - spotify_key) % 12
#     key_error_score = min(key_dist, 12 - key_dist) / 6.0
#     extra = pd.DataFrame([{
#         'Feature': 'key_error_score',
#         'Your Algorithm': key_error_score,
#         'Spotify': None,
#         'Difference': None
#     }])
#     comparison = pd.concat([comp_df, extra], ignore_index=True)

#     # save and log
#     base_filename = f"comparison_{title}_by_{artist}.csv"
#     output_path = COMPARISONS_FOLDER / base_filename
#     try:
#         comparison.to_csv(output_path, index=False)
#         logging.info(f"Comparison saved to: {output_path}")
#     except Exception as e:
#         logging.error(f"Error saving comparison for {title} by {artist}: {e}")

# --- Core Classes ---
class SoundCloudScraper:
    """Handles searching SoundCloud and parsing results using Selenium."""

    def __init__(self):
        self.driver = None
        self.session = requests.Session() # Keep requests for potential future use
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.BANNED_KEYWORDS = [
            "remastered", "live", "album version", "mono", "slowed", 
            "reverb", "edit", "clean", "explicit", "version"
        ]

    def _setup_driver(self):
        """Initializes the Selenium WebDriver for this scraper instance."""
        if self.driver:
             return True # Already initialized
        try:
            logging.info("Setting up Selenium WebDriver for SoundCloudScraper...")
            chrome_options = Options()
            # chrome_options.add_argument("--headless")  # Consider headless for performance
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument(f'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

            # Ensure Chrome binary is found (especially on macOS/Linux)
            # try:
            #     chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" # Example for macOS
            # except:
            #     logging.warning("Default Chrome binary location not found or specified. Assuming it's in PATH.")
            #     pass

            service = ChromeService(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(5)
            logging.info("SoundCloudScraper WebDriver setup complete.")
            return True
        except Exception as e:
            logging.error(f"Failed to setup SoundCloudScraper WebDriver: {e}")
            self.driver = None
            return False
            
    def _quit_driver(self):
        """Closes the Selenium WebDriver if it was initialized."""
        if self.driver:
            logging.info("Closing SoundCloudScraper WebDriver.")
            try:
                self.driver.quit()
            except Exception as e:
                logging.warning(f"Error quitting SoundCloudScraper WebDriver: {e}")
            self.driver = None

    def search(self, song_name, artist_name):
        """Searches SoundCloud using Selenium to handle dynamic content."""
        if not self._setup_driver():
            return None

        query = f"{song_name} {artist_name}"
        search_url = SOUNDCLOUD_SEARCH_URL.format(query=quote_plus(query))
        logging.info(f"Searching SoundCloud via Selenium for: '{query}' at {search_url}")
        
        page_source = None
        try:
            self.driver.get(search_url)
            
            # Wait for the main results list container to be present in the DOM
            results_list_selector = (By.CSS_SELECTOR, "ul.lazyLoadingList__list")
            logging.info(f"Waiting up to {SELENIUM_TIMEOUT}s for results list ({results_list_selector[1]}) to load...")
            wait = WebDriverWait(self.driver, SELENIUM_TIMEOUT)
            wait.until(EC.presence_of_element_located(results_list_selector))
            logging.info("Results list found.")
            
            # Optional: Add a small explicit wait to allow content within the list to potentially load
            time.sleep(2)
            
            page_source = self.driver.page_source
            
        except TimeoutException:
            logging.error(f"Timeout waiting for SoundCloud search results list ({results_list_selector[1]}) to appear for query '{query}'.")
        except Exception as e:
            logging.error(f"Selenium search request failed for '{query}': {e}")
        # Removed finally block with _quit_driver here - driver should persist until pipeline is done
        # Consider quitting driver in the pipeline logic after processing all songs

        return page_source

    def parse_results(self, html_content):
        """Parses SoundCloud search results HTML."""
        if not html_content:
            return []
            
        soup = BeautifulSoup(html_content, 'lxml')
        search_results = []
        
        # *** Updated Selectors based on new HTML structure (April 2025) ***
        # Find the main list containing search results
        results_list = soup.find('ul', class_=lambda x: x and 'lazyLoadingList__list' in x.split())
        if not results_list:
             logging.warning("Could not find the main search results list (ul.lazyLoadingList__list). Structure might have changed again.")
             return []

        # Find all individual search item list elements within the main list
        search_items = results_list.find_all('li', class_=lambda x: x and 'searchList__item' in x.split(), recursive=False)
        
        if not search_items:
            logging.warning("Could not find individual search items (li.searchList__item). Structure might have changed.")
            return []

        logging.info(f"Found {len(search_items)} potential search items (li elements) in search results.")

        for item in search_items:
            # Focus only on track items (div with class 'sound' and 'track')
            track_item = item.find('div', class_=lambda x: x and 'sound' in x.split() and 'track' in x.split())
            if not track_item:
                logging.debug("Skipping item: Not a track item.")
                continue

            try:
                # Find title link and URL
                title_link_element = track_item.find('a', class_=lambda x: x and 'soundTitle__title' in x.split())
                if not title_link_element or not title_link_element.get('href'):
                    logging.warning("Skipping track item: Missing title link or href.")
                    continue
                
                track_url_path = title_link_element['href']
                # Basic validation for a track URL path
                if not track_url_path or not track_url_path.startswith('/') or '/sets/' in track_url_path or '/people/' in track_url_path:
                     logging.warning(f"Skipping track item: Invalid or non-track URL path '{track_url_path}'.")
                     continue

                track_url = f"https://soundcloud.com{track_url_path}"
                
                # Find the actual title text within the link (often in a span)
                title_span = title_link_element.find('span', recursive=False) # Check immediate span first
                title = title_span.text.strip() if title_span else title_link_element.text.strip()


                # Find artist link/name
                artist_link_element = track_item.find('a', class_=lambda x: x and 'soundTitle__username' in x.split())
                artist_span = artist_link_element.find('span', class_='soundTitle__usernameText') if artist_link_element else None
                artist = artist_span.text.strip() if artist_span else (artist_link_element.text.strip() if artist_link_element else "Unknown Artist")
                
                if title and artist != "Unknown Artist" and track_url:
                    search_results.append({
                        'title': title,
                        'artist': artist,
                        'url': track_url
                    })
                    logging.debug(f"Successfully parsed track: Title='{title}', Artist='{artist}', URL='{track_url}'")
                else:
                     logging.warning(f"Skipping track item: Missing title, artist, or URL after parsing. Title:'{title}', Artist:'{artist}', URL:'{track_url}'")

            except Exception as e:
                logging.error(f"Error parsing a track search result item: {e}", exc_info=True) # Log traceback for errors
                continue
                
        logging.info(f"Parsed {len(search_results)} valid tracks from search results.")
        return search_results

    def find_best_match(self, search_results, target_song, target_artist):
        """Filters search results to find the best match based on fuzzy scoring."""
        best_match = None
        highest_score = -1

        logging.info(f"Filtering {len(search_results)} results for '{target_song}' by '{target_artist}'")

        for result in search_results:
            if result['title'] in self.BANNED_KEYWORDS or result['artist'] in self.BANNED_KEYWORDS:
                continue
                
            if result['title'] == target_song and result['artist'] == target_artist:
                logging.info(f"Found exact match: '{result['title']}' by '{result['artist']}'")
                return result

            title_score = fuzz.ratio(target_song.lower(), result['title'].lower())
            # Partial ratio can be good if target name is part of a longer title
            title_score_partial = fuzz.partial_ratio(target_song.lower(), result['title'].lower())
            # Use the higher of the two title scores
            effective_title_score = max(title_score, title_score_partial)

            artist_score = fuzz.ratio(target_artist.lower(), result['artist'].lower())
            
            # Weighted score: prioritize artist match slightly more
            # Adjust weights as needed
            combined_score = (effective_title_score * 0.4) + (artist_score * 0.6)

            logging.debug(f"  Candidate: '{result['title']}' by '{result['artist']}' "
                          f"(Title Score: {effective_title_score}, Artist Score: {artist_score}, Combined: {combined_score:.2f}) "
                          f"URL: {result['url']}")


            # Prefer perfect artist matches if scores are close
            if combined_score > highest_score:
                # Check if this is a significantly better score OR if the artist is a much better match
                 is_better_artist = artist_score > fuzz.ratio(target_artist.lower(), best_match['artist'].lower()) if best_match else True
                 # Require a significant score improvement or a much better artist match to switch
                 if combined_score > highest_score + 5 or (combined_score >= highest_score and is_better_artist) : 
                    highest_score = combined_score
                    best_match = result
                    logging.debug(f"    New best candidate found.")


        if best_match and highest_score >= MATCH_THRESHOLD:
            logging.info(f"Best match found: '{best_match['title']}' by '{best_match['artist']}' "
                         f"with score {highest_score:.2f}")
            return best_match
        else:
            logging.warning(f"No suitable match found for '{target_song}' by '{target_artist}' "
                            f"(Highest score: {highest_score:.2f}, Threshold: {MATCH_THRESHOLD})")
            return None

# --- New YTDLPDownloader Class ---
class YTDLPDownloader:
    """Handles downloading audio tracks using the yt-dlp library."""

    def __init__(self, download_folder):
        self.download_folder = Path(download_folder)

    def download_track(self, url, expected_artist, expected_title, output_path=None):
        """Downloads a single track from the given URL using yt-dlp."""
        
        logging.info(f"Processing download for URL: {url}")
        download_successful = False
        final_filename = None
        final_filename_path = None # Store the full path for checking
        
        # --- Step 1: Extract Info --- 
        try:
            # Basic options just for extracting info
            ydl_opts_info = {
                'quiet': True,
                'logger': logging.getLogger('yt-dlp'),
                'noprogress': True,
                'noplaylist': True,
                 # Add cookie file if needed for restricted content (requires browser addon like Get cookies.txt LOCALLY)
                 # 'cookiefile': 'path/to/your/cookies.txt',
            }
            with yt_dlp.YoutubeDL(ydl_opts_info) as ydl_info:
                logging.debug("Extracting metadata with yt-dlp...")
                info_dict = ydl_info.extract_info(url, download=False) 
            
            if not info_dict:
                logging.error(f"yt-dlp could not extract info for URL: {url}")
                return None, False
            
            # --- Step 2: Determine Output Filename --- 
            # Use sanitize_filename on extracted title/artist if available for better template matching
            extracted_artist = sanitize_filename(info_dict.get('artist', info_dict.get('uploader', expected_artist)))
            extracted_title = sanitize_filename(info_dict.get('title', expected_title))
            
            # Construct the final base filename (assuming mp3 postprocessing)
            final_filename_base = f'{extracted_artist} - {extracted_title}.mp3'
            final_filename_path = self.download_folder / final_filename_base
            logging.info(f"Determined target filename: {final_filename_path.name}")

            # --- Step 3: Check if File Already Exists --- 
            if final_filename_path.exists():
                 logging.info(f"MP3 file '{final_filename_path.name}' already exists. Skipping download.")
                 return str(final_filename_path), True # Return path and success

        except yt_dlp.utils.DownloadError as e:
            # Handle errors during info extraction (e.g., video unavailable)
            logging.error(f"yt-dlp info extraction error for {url}: {e}")
            return None, False
        except Exception as e:
            logging.error(f"Unexpected error during info extraction for {url}: {e}", exc_info=True)
            return None, False

        # --- Step 4: Download if file doesn't exist --- 
        try:
            # Define full download options with finalized output template
            ydl_opts_download = {
                'format': 'bestaudio/best',
                # Use the finalized path (without extension, yt-dlp adds it)
                'outtmpl': str(final_filename_path.with_suffix('.%(ext)s')),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'noplaylist': True,
                'logger': logging.getLogger('yt-dlp'),
                'noprogress': True,
                'retries': 3,
                'fragment_retries': 3,
                # Add cookie file if needed
                # 'cookiefile': 'path/to/your/cookies.txt',
            }
            
            # If output_path is provided, use it for yt-dlp's outtmpl
            if output_path is not None:
                ydl_opts_download['outtmpl'] = str(Path(output_path).with_suffix('.%(ext)s'))
            
            logging.info(f"Attempting download via yt-dlp with options: {ydl_opts_download}")
            with yt_dlp.YoutubeDL(ydl_opts_download) as ydl_download:
                ydl_download.download([url])

            # --- Step 5: Verify Download --- 
            if final_filename_path.exists():
                logging.info(f"yt-dlp download successful. File created: {final_filename_path}")
                download_successful = True
                final_filename = str(final_filename_path)
            else:
                # Check if maybe extension is different (less likely but possible)
                base_name = final_filename_path.stem
                found_files = list(self.download_folder.glob(f'{re.escape(base_name)}.*'))
                if found_files:
                    logging.info(f"yt-dlp download likely successful. Found file: {found_files[0]}")
                    download_successful = True
                    final_filename = str(found_files[0])
                else:
                    logging.error(f"yt-dlp download finished, but expected output file not found: {final_filename_path.name}")

        except yt_dlp.utils.DownloadError as e:
            logging.error(f"yt-dlp download error for {url}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during yt-dlp download for {url}: {e}", exc_info=True)
            
        return final_filename, download_successful

class SpotifyFeaturesTunable:
    
    def __init__(
        self,
        sample_rate: int = 22050,
        tempo_range: tuple = (60.0, 180.0),
        weights: dict = None
    ):
        self.sample_rate = sample_rate
        self.tempo_min, self.tempo_max = tempo_range
        warnings.filterwarnings('ignore')
        logging.basicConfig(level=logging.INFO)

        # Default tunable weights
        default = {
            'danceability':     {'beat_reg': 0.4, 'bass': 0.3, 'pulse': 0.3},
            'energy':           {'rms': 0.6, 'entropy': 0.2, 'dyn_range': 0.2},
            'acousticness':     {'harmonic_ratio': 0.4, 'centroid': 0.3, 'flatness': 0.2, 'contrast': 0.1},
            'valence':          {'mode': 0.3, 'energy': 0.2, 'tempo': 0.2, 'brightness': 0.15, 'rhythm': 0.15},
            'instrumentalness': {'mfcc_var': 0.6, 'pitch_var': 0.4},
            'speechiness':      {'zcr': 0.4, 'rhythm': 0.3, 'mfcc': 0.3},
            'liveness':         {'dyn_range': 0.4, 'high_freq': 0.3, 'decay': 0.3},
            'tempo':            {'scale': 1.0},
            'loudness':         {'scale': 1.0},
            'key':              {'weight': 1.0}
        }
        self.weights = weights or default

    def _normalize(self, value, min_val, max_val, new_min, new_max):
        v = np.clip(value, min_val, max_val)
        return (v - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

    def precompute_base_features(self, file_path: str) -> dict:
        """Runs all expensive librosa computations once and returns raw features."""
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        y_h, y_p = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(y=y_p, sr=sr)
        tempo_raw, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Base features
        times = librosa.frames_to_time(beats, sr=sr)
        beat_reg = 1 - min(1., np.std(np.diff(times)) / np.mean(np.diff(times))) if len(times) > 1 else 0.0
        spec = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        bass_raw = np.mean(spec[freqs <= 250]) / (np.mean(spec) + 1e-8)
        pulse_raw = librosa.feature.rms(y=y_p)[0].mean()
        
        rms = librosa.feature.rms(y=y)[0]
        st = np.abs(librosa.stft(y))
        stn = st / (np.sum(st, axis=0, keepdims=True) + 1e-8)
        entropy_raw = -np.sum(stn * np.log2(stn + 1e-8), axis=0).mean()
        dyn_range_raw = np.percentile(rms, 95) / (np.percentile(rms, 10) + 1e-8)
        
        centroid_raw = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
        flatness_raw = librosa.feature.spectral_flatness(y=y).mean()
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        hr = np.mean(librosa.feature.rms(y=y_h)[0])
        pr = np.mean(librosa.feature.rms(y=y_p)[0])
        harmonic_ratio_raw = hr / (hr + pr + 1e-8)
        contrast_ratio_raw = (contrast[:2].mean() / (contrast[-2:].mean() + 1e-8)) if contrast.shape[0] >= 6 else 0.5
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_var_raw = np.var(mfcc[2:8, :], axis=1).mean()
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pmax = np.array([pitches[np.argmax(mags[:, i]), i] for i in range(mags.shape[1]) if mags[:, i].max() > 0])
        pitch_var_raw = np.std(pmax) / (np.mean(pmax) + 1e-8) if len(pmax) > 0 else 0
        
        zcr_raw = librosa.feature.zero_crossing_rate(y=y)[0].mean()
        mfcc_delta_var_raw = np.var(librosa.feature.delta(mfcc), axis=1).mean()
        
        dyn_range_liveness_raw = np.percentile(rms, 95) - np.percentile(rms, 10)
        high_freq_raw = np.percentile(librosa.feature.spectral_rolloff(y=y, sr=sr)[0], 90)
        seg = librosa.util.frame(onset_env, frame_length=10, hop_length=1)
        slope = stats.linregress(np.arange(seg.shape[1]), seg.mean(axis=0))[0] if seg.shape[1] > 1 else 0
        decay_raw = 1 - self._normalize(abs(slope), 0.001, 0.1, 0, 1)

        chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, bins_per_octave=36, n_chroma=12)
        chroma_smooth = np.minimum(1.0, librosa.decompose.nn_filter(chroma, aggregate=np.median, metric='cosine'))
        key_profile = chroma_smooth.sum(axis=1)

        return {
            'tempo_raw': tempo_raw, 'beat_reg': beat_reg, 'bass_raw': bass_raw, 'pulse_raw': pulse_raw,
            'rms_mean': rms.mean(), 'entropy_raw': entropy_raw, 'dyn_range_raw': dyn_range_raw,
            'harmonic_ratio_raw': harmonic_ratio_raw, 'centroid_raw': centroid_raw, 'flatness_raw': flatness_raw,
            'contrast_ratio_raw': contrast_ratio_raw, 'onset_env_mean': onset_env.mean(), 'rms_db_mean': librosa.amplitude_to_db(rms, ref=1.0).mean(),
            'mfcc_var_raw': mfcc_var_raw, 'pitch_var_raw': pitch_var_raw, 'zcr_raw': zcr_raw,
            'mfcc_delta_var_raw': mfcc_delta_var_raw, 'dyn_range_liveness_raw': dyn_range_liveness_raw,
            'high_freq_raw': high_freq_raw, 'decay_raw': decay_raw, 'key_profile': key_profile
        }

    def compute_from_precomputed(self, base_feats: dict) -> dict:
        """Computes final features from precomputed values using current weights."""
        w_d = self.weights['danceability']
        dance = w_d['beat_reg'] * base_feats['beat_reg'] + \
                w_d['bass'] * base_feats['bass_raw'] + \
                w_d['pulse'] * self._normalize(base_feats['pulse_raw'], 0, 0.2, 0, 1)
        
        w_e = self.weights['energy']
        energy = w_e['rms'] * self._normalize(base_feats['rms_mean'], 0, 0.2, 0, 1) + \
                 w_e['entropy'] * self._normalize(base_feats['entropy_raw'], 0, 5, 0, 1) + \
                 w_e['dyn_range'] * self._normalize(base_feats['dyn_range_raw'], 1, 20, 0, 1)

        w_a = self.weights['acousticness']
        ac = w_a['harmonic_ratio'] * base_feats['harmonic_ratio_raw'] + \
             w_a['centroid'] * (1 - self._normalize(base_feats['centroid_raw'], 500, 3000, 0, 1)) + \
             w_a['flatness'] * (1 - base_feats['flatness_raw']) + \
             w_a['contrast'] * self._normalize(base_feats['contrast_ratio_raw'], 0.5, 5, 0, 1)

        tempo = float(np.clip(base_feats['tempo_raw'], self.tempo_min, self.tempo_max))
        w_v = self.weights['valence']
        val = w_v['mode'] * 0.7 + \
              w_v['energy'] * energy + \
              w_v['tempo'] * self._normalize(tempo, self.tempo_min, self.tempo_max, 0, 1) + \
              w_v['brightness'] * self._normalize(base_feats['centroid_raw'], 500, 3000, 0, 1) + \
              w_v['rhythm'] * self._normalize(base_feats['onset_env_mean'], 0, 0.5, 0, 1)

        w_i = self.weights['instrumentalness']
        inst = 1 - (w_i['mfcc_var'] * self._normalize(base_feats['mfcc_var_raw'], 0.1, 5, 0, 1) + \
                    w_i['pitch_var'] * self._normalize(base_feats['pitch_var_raw'], 0, 0.5, 0, 1))

        w_s = self.weights['speechiness']
        sp = w_s['zcr'] * self._normalize(base_feats['zcr_raw'], 0.05, 0.15, 0, 1) + \
             w_s['rhythm'] * (1 - base_feats['beat_reg']) + \
             w_s['mfcc'] * self._normalize(base_feats['mfcc_delta_var_raw'], 0.5, 5, 0, 1)

        w_l = self.weights['liveness']
        live = w_l['dyn_range'] * self._normalize(base_feats['dyn_range_liveness_raw'], 0.01, 0.1, 0, 1) + \
               w_l['high_freq'] * self._normalize(base_feats['high_freq_raw'], 3000, 8000, 0, 1) + \
               w_l['decay'] * base_feats['decay_raw']

        profile = base_feats['key_profile']
        profile /= profile.sum() + 1e-8
        major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        major /= major.sum(); minor /= minor.sum()
        cor_maj = [np.corrcoef(profile, np.roll(major, i))[0, 1] for i in range(12)]
        cor_min = [np.corrcoef(profile, np.roll(minor, i))[0, 1] for i in range(12)]
        key = int(np.argmax(cor_maj)) if max(cor_maj) > max(cor_min) else int(np.argmax(cor_min))

        return {
            'danceability': float(np.clip(dance, 0, 1)), 'energy': float(np.clip(energy, 0, 1)),
            'acousticness': float(np.clip(ac, 0, 1)), 'valence': float(np.clip(val, 0, 1)),
            'tempo': self.weights['tempo']['scale'] * tempo,
            'loudness': self.weights['loudness']['scale'] * float(np.clip(base_feats['rms_db_mean'], -60, 0)),
            'instrumentalness': float(np.clip(inst, 0, 1)), 'speechiness': float(np.clip(sp, 0, 1)),
            'liveness': float(np.clip(live, 0, 1)), 'key': key
        }

    def extract_features(self, file_path: str) -> dict:
        base_feats = self.precompute_base_features(file_path)
        return self.compute_from_precomputed(base_feats)

    def flatten_weights(self):
        flat, keys = [], []
        for feat, sub in self.weights.items():
            for k, v in sub.items():
                flat.append(v)
                keys.append((feat, k))
        return np.array(flat), keys

    def unflatten_weights(self, flat, keys):
        new = {}
        for (feat, k), v in zip(keys, flat):
            new.setdefault(feat, {})[k] = v
        self.weights = new
    
    def analyze_track(self, file_path):
        """Analyze a track and return Spotify-like audio features."""
        try:
            features = self.extract_features(file_path)
            return features
        except Exception as e:
            logging.error(f"Error analyzing track: {e}")
            return None

    def save_features_to_cache(self, file_path: str, features_dict):
        row = {"file_path": file_path}
        row.update(features_dict)

        # Create or append to the CSV
        if os.path.exists(FEATURE_CACHE):
            df = pd.read_csv(FEATURE_CACHE)
            if file_path in df['file_path'].values:
                print(f"Features already cached for: {file_path}")
                return
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        
        df.to_csv(FEATURE_CACHE, index=False)
        print(f"Saved features for {file_path}")

    def get_features_from_cache(self, file_path: str):
        if not os.path.exists(FEATURE_CACHE):
            return None
        
        df = pd.read_csv(FEATURE_CACHE)
        match = df[df['file_path'] == file_path]
        
        if match.empty:
            return None
        else:
            return match.iloc[0].to_dict()

class SoundCloudPipeline:
    """Orchestrates the entire SoundCloud song downloading process using yt-dlp."""

    def __init__(self, download_folder=DOWNLOAD_FOLDER, checkpoint_file=CHECKPOINT_FILE, start_index=0, end_index=100):
        self.download_folder = Path(download_folder)
        self.checkpoint_file = Path(checkpoint_file)
        self.scraper = SoundCloudScraper()
        self.downloader = YTDLPDownloader(self.download_folder) # Use the new downloader
        self.checkpoint_data = self._load_checkpoint()
        self.song_list = self.get_songs_from_file('./music_info_cleaned.csv', start_index, end_index)
        self.analyzer = SpotifyFeaturesTunable()
        self.downloaded_songs_paths = []
        # Ensure download folder exists
        self.download_folder.mkdir(parents=True, exist_ok=True)

        # Load Spotify ground-truth baseline
        self.baseline = (
            pd.read_csv(SPOTIFY_BASELINE)
              .set_index(['name', 'artist'])
        )

        logging.info(f"Using download folder: {self.download_folder.resolve()}")
        logging.info(f"Using checkpoint file: {self.checkpoint_file.resolve()}")

    
    def _load_checkpoint(self):
        """Loads checkpoint data from the JSON file."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                try:
                
                    data = json.load(f)
                    logging.info(f"Loaded {len(data)} entries from checkpoint file.")
                    # Compatibility check: Ensure entries have necessary keys
                    cleaned_data = {}
                    for url, entry in data.items():
                        if all(k in entry for k in ['song_name', 'artist_name', 'soundcloud_url', 'download_status']):
                            cleaned_data[url] = entry
                        else:
                            logging.warning(f"Skipping malformed checkpoint entry for URL: {url}")
                    return cleaned_data
                except (json.JSONDecodeError, IOError) as e:
                    logging.error(f"Error loading checkpoint file {self.checkpoint_file}: {e}. Starting fresh.")
                    return {}
        else:
            logging.info("Checkpoint file not found. Starting fresh.")
            return {}

    def _save_checkpoint(self):
        """Saves the current checkpoint data to the JSON file."""
        try:
            # Ensure directory exists before writing
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_data, f, indent=4, ensure_ascii=False)
            logging.debug(f"Checkpoint data saved to {self.checkpoint_file}")
        except IOError as e:
            logging.error(f"Error saving checkpoint file {self.checkpoint_file}: {e}")

    def _is_downloaded(self, filename):
         """Checks if a file with the expected name already exists (more flexible check)."""
         # Check for exact filename (often mp3)
         if (self.download_folder / filename).exists():
              return True
         # Check if a file with the same base name but different extension exists
         base_name = Path(filename).stem
         found_files = list(self.download_folder.glob(f'{base_name}.*'))
         if found_files:
              logging.info(f"Found existing file matching base name '{base_name}': {found_files[0].name}")
              return True
         return False

    def get_songs_from_file(self, file_path, start_index=0, end_index=100):
        """Reads a CSV file with song names and artists, and returns a list of dictionaries."""
        df = pd.read_csv(file_path)
        results = df[['name', 'artist']][start_index:end_index].drop_duplicates()
        return results.to_dict(orient='records')

    def process_song(self, song_name, artist_name):
        """Processes a single song: search, filter, checkpoint, download via yt-dlp."""
        logging.info(f"--- Processing: '{song_name}' by '{artist_name}' ---")
        
        # Checkpoint logic: Find existing entry based on song/artist name
        soundcloud_url = None
        existing_entry = None
        for url, data in self.checkpoint_data.items():
            if data.get('song_name', '').lower() == song_name.lower() and \
               data.get('artist_name', '').lower() == artist_name.lower():
                soundcloud_url = url
                existing_entry = data
                logging.info(f"Found existing entry in checkpoint for '{song_name}' - URL: {soundcloud_url}")
                break

        if not soundcloud_url:
            # 1. Search SoundCloud if not in checkpoint
            search_html = self.scraper.search(song_name, artist_name)
            if not search_html:
                logging.error("Failed to get SoundCloud search results.")
                return # Skip this song

            # 2. Parse and Filter Results
            search_results = self.scraper.parse_results(search_html)
            best_match = self.scraper.find_best_match(search_results, song_name, artist_name)

            if not best_match:
                logging.error("Could not find a suitable match on SoundCloud.")
                # Optionally record failure in checkpoint?
                return # Skip this song

            soundcloud_url = best_match['url']
            logging.info(f"Selected SoundCloud URL: {soundcloud_url}")

            # 3. Save Checkpoint (SoundCloud URL found)
            # Use the URL itself as the key
            self.checkpoint_data[soundcloud_url] = {
                'song_name': song_name,
                'artist_name': artist_name,
                'matched_title': best_match['title'],
                'matched_artist': best_match['artist'],
                'soundcloud_url': soundcloud_url,
                'download_status': 'pending', # Initial status
                'output_file': None
            }
            existing_entry = self.checkpoint_data[soundcloud_url] # Update existing_entry reference
            self._save_checkpoint()
        
        # --- Proceed to Download with yt-dlp ---
        if not existing_entry: # Should not happen if URL was found/added, but safety check
             logging.error("Logic error: No checkpoint entry available for download.")
             return

        # Check download status in checkpoint
        current_status = existing_entry.get('download_status', 'pending')
        output_file = existing_entry.get('output_file')

        if current_status == 'completed' and output_file and self._is_downloaded(output_file):
            logging.info(f"Checkpoint indicates already downloaded and file exists: '{output_file}'. Skipping.")
            self.downloaded_songs_paths.append(DOWNLOAD_FOLDER / Path(output_file).name)
            return

        
        logging.info(f"Attempting download for {soundcloud_url}...")
        
        # Always use dataset's name/artist for filename
        sanitized_artist = sanitize_filename(artist_name)
        sanitized_title = sanitize_filename(song_name)
        final_filename = f"{sanitized_artist} - {sanitized_title}.mp3"
        final_filepath = self.download_folder / final_filename

        # Check if file already exists
        if final_filepath.exists():
            logging.info(f"File already exists: {final_filepath}, skipping download.")
            self.downloaded_songs_paths.append(final_filepath)
            # Update checkpoint as completed if not already
            # (optional: update checkpoint_data here)
            return

        # Download using YTDLPDownloader, forcing output_path
        final_filename_str, download_successful = self.downloader.download_track(
            soundcloud_url,
            artist_name,
            song_name,
            output_path=final_filepath
        )

        if download_successful and final_filename_str:
            existing_entry['download_status'] = 'completed'
            existing_entry['output_file'] = Path(final_filename_str).name
            self.downloaded_songs_paths.append(final_filepath)
        else:
            existing_entry['download_status'] = 'failed_ytdlp'
            existing_entry['output_file'] = None
            self.downloaded_songs_paths.append("failed")
        self._save_checkpoint()
        logging.info(f"--- Finished processing: '{song_name}' by '{artist_name}' (Status: {existing_entry['download_status']}) ---")

    def download_songs(self):
        """Runs the pipeline for a list of songs."""
        logging.info(f"Starting pipeline for {len(self.song_list)} songs...")
        total_processed = 0
        logging.info("Song list in download_songs:\n " + str(self.song_list[:10]))
        try:
            for i, song_info in enumerate(self.song_list):
                song_name = song_info.get('name')
                artist_name = song_info.get('artist')

                if not song_name or not artist_name:
                    logging.warning(f"Skipping item {i+1}: Missing 'name' or 'artist'. Data: {song_info}")
                    continue

                logging.info(f"[download_songs] Processing song {i+1}/{len(self.song_list)}: '{song_name}' by '{artist_name}'")
                self.process_song(song_name, artist_name)
                total_processed += 1
                logging.info(f"[download_songs] Completed {total_processed}/{len(self.song_list)}")

        finally:
            # Ensure Selenium driver for scraper is closed when pipeline finishes or errors out
            logging.info("Pipeline run finished. Cleaning up SoundCloudScraper driver...")
            self.scraper._quit_driver()
            logging.info("Cleanup complete.")
        logging.info("[download_songs] All downloads attempted. Proceeding to next steps.")

    def save_tuning_csv(self, output_csv: str):
        """
        Build a DataFrame of predicted vs. Spotify features for each downloaded song,
        then append to or create the CSV at output_csv.
        """
        logging.info("[save_tuning_csv] Starting feature extraction and CSV saving.")
        # Collect rows for each song
        records = []
        # Build a normalized lookup for the baseline
        norm_baseline = {(normalize(name), normalize(artist)): row for (name, artist), row in self.baseline.iterrows()}
        for idx, path in enumerate(self.downloaded_songs_paths):
            # Skip failed entries
            if isinstance(path, str):
                if path == "failed":
                    continue
                path = Path(path)
            basename = path.stem
            try:
                artist, title = basename.split(' - ', 1)
            except ValueError:
                continue
            logging.info(f"[save_tuning_csv] ({idx+1}/{len(self.downloaded_songs_paths)}) Extracting features for: '{title}' by '{artist}'")
            # Extract model predictions
            feats = self.analyzer.analyze_track(str(path))
            # Normalize for lookup
            norm_key = (normalize(title), normalize(artist))
            obs_row = norm_baseline.get(norm_key)
            if obs_row is None:
                logging.warning(f"No baseline entry for: {title} by {artist}, skipping.")
                continue
            obs = obs_row.to_dict()
            row = {'file_path': str(path), 'name': title, 'artist': artist}
            for feature, val in feats.items():
                row[feature] = val
                row[f"{feature}_spotify"] = obs.get(feature)
            records.append(row)
        # Create DataFrame and append to CSV
        df = pd.DataFrame(records)
        write_header = not os.path.exists(output_csv)
        df.to_csv(
            output_csv,
            mode='a',
            header=write_header,
            index=False
        )
        action = 'written to' if write_header else 'appended to'
        logging.info(f"[save_tuning_csv] Tuning data {action} {output_csv} with {len(df)} records.")
        logging.info("[save_tuning_csv] Feature extraction and CSV saving complete.")
   
class HyperparameterTuner:
    """
    Optimizes SpotifyFeaturesTunable weights with a train/validation split.
    """
    def __init__(
        self,
        model,
        full_baseline_df: pd.DataFrame,
        val_frac: float = 0.2,
        seed: int = 42
    ):
        """
        model: an instance of SpotifyFeaturesTunable
        full_baseline_df: DataFrame with columns 'file_path' plus Spotify ground-truth features
        val_frac: fraction of data reserved for validation
        seed: for reproducible train/validation split
        """
        self.model = model
        # Shuffle and split
        df = full_baseline_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        split_idx = int(len(df) * (1 - val_frac))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        # Index by file_path for easy lookup
        self.train_baseline = train_df.set_index('file_path')
        self.val_baseline = val_df.set_index('file_path')
        self.train_tracks = self.train_baseline.index.tolist()
        self.val_tracks = self.val_baseline.index.tolist()
        logging.info(f"[HyperparameterTuner] Initialized with {len(self.train_tracks)} training tracks and {len(self.val_tracks)} validation tracks.")
        
        # Pre-compute and cache base features
        self.feature_cache = {}
        logging.info("[HyperparameterTuner] Pre-computing base features for all tracks...")
        all_tracks = self.train_tracks + self.val_tracks
        for i, fp in enumerate(all_tracks):
            logging.info(f"[HyperparameterTuner] Pre-computing features for track {i+1}/{len(all_tracks)}: {fp}")
            self.feature_cache[fp] = self.model.precompute_base_features(fp)
        logging.info("[HyperparameterTuner] Base feature pre-computation complete.")


    def _objective(self, flat_weights):
        """
        Objective on training set: mean squared error for continuous features
        plus classification penalty for key.
        """
        # Unpack weights into model
        _, keys = self.model.flatten_weights()
        self.model.unflatten_weights(flat_weights, keys)

        errors = []
        # Continuous feature names
        cont_feats = [
            'danceability','energy','acousticness','valence',
            'tempo','loudness','instrumentalness','speechiness','liveness'
        ]
        for fp in self.train_tracks:
            base_feats = self.feature_cache[fp]
            pred = self.model.compute_from_precomputed(base_feats)
            obs = self.train_baseline.loc[fp]
            # MSE for continuous features
            for f in cont_feats:
                errors.append((pred[f] - obs[f])**2)
            # Key classification penalty
            key_weight = self.model.weights['key']['weight']
            errors.append(key_weight * (0 if pred['key']==int(obs['key']) else 1))
        mean_err = float(np.mean(errors))
        logging.debug(f"[HyperparameterTuner] Objective evaluated: mean error = {mean_err}")
        return mean_err

    def tune(self, maxiter: int = 50, popsize: int = 15, tol: float = 1e-5):
        """
        Run differential evolution on the training set.
        Returns the OptimizeResult and logs training loss.
        """
        logging.info("[HyperparameterTuner] Starting hyperparameter tuning...")
        init_vec, _ = self.model.flatten_weights()
        bounds = [(0.0, 1.0)] * len(init_vec)

        result = differential_evolution(
            self._objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol
        )
        # Apply best weights to model
        _, keys = self.model.flatten_weights()
        self.model.unflatten_weights(result.x, keys)
        logging.info(f"[HyperparameterTuner] Tuning complete. Training loss: {result.fun}")
        return result

    def validate(self):
        """
        Compute and return loss on the validation set using the tuned weights.
        """
        logging.info("[HyperparameterTuner] Starting validation on held-out set...")
        errors = []
        cont_feats = [
            'danceability','energy','acousticness','valence',
            'tempo','loudness','instrumentalness','speechiness','liveness'
        ]
        for fp in self.val_tracks:
            base_feats = self.feature_cache[fp]
            pred = self.model.compute_from_precomputed(base_feats)
            obs = self.val_baseline.loc[fp]
            # MSE for continuous features
            for f in cont_feats:
                errors.append((pred[f] - obs[f])**2)
            # Key penalty
            key_weight = self.model.weights['key']['weight']
            errors.append(key_weight * (0 if pred['key']==int(obs['key']) else 1))

        val_loss = float(np.mean(errors))
        logging.info(f"[HyperparameterTuner] Validation complete. Validation loss: {val_loss}")
        return val_loss

