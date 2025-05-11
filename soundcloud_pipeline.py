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

# --- Configuration ---
DOWNLOAD_FOLDER = Path("downloads")
CHECKPOINT_FILE = DOWNLOAD_FOLDER / "song_url.json"
COMPARISONS_FOLDER = Path("comparisons")
SOUNDCLOUD_SEARCH_URL = "https://soundcloud.com/search?q={query}"
FEATURE_CACHE = "./audio_features_cache.csv"

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


def compare_results(results, song_name, artist_name):
    # Load Spotify data from CSV
    spotify_data = pd.read_csv('./music_info_cleaned.csv')

    spotify_song = spotify_data[(spotify_data['name'].str.lower() == song_name.lower()) & (spotify_data['artist'].str.lower() == artist_name.lower())]
    
    if len(spotify_song) == 0:
        logging.info(f"THIS DIDNOT WORKOUT: {artist_name} {song_name}")
        return

    spotify_song = spotify_song.iloc[0]

    base_filename = f"comparison_{spotify_song['name']}_by_{spotify_song['artist']}.csv"
    
    #check if the comparison exist already
    if base_filename in os.listdir(COMPARISONS_FOLDER):
        logging.info("The song has already been compared, skipping...")
        return

    

    

    logging.info("we got the spotify song, creating comparison dataframe now:\n"+ str(spotify_song))
    
    # Calculate key error score
    predicted_key = results['key']
    spotify_key = spotify_song['key']
    key_distance = abs(predicted_key - spotify_key) % 12
    key_error_score = min(key_distance, 12 - key_distance) / 6.0  # Normalize to [0, 1]

    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Feature': ['danceability', 'energy', 'acousticness', 'instrumentalness', 
                    'liveness', 'valence', 'speechiness', 'tempo', 'loudness', 'key'],
        'Your Algorithm': [results['danceability'], results['energy'], results['acousticness'],
                        results['instrumentalness'], results['liveness'], results['valence'],
                        results['speechiness'], results['tempo'], results['loudness'], results['key']],
        'Spotify': [spotify_song['danceability'], spotify_song['energy'], spotify_song['acousticness'],
                spotify_song['instrumentalness'], spotify_song['liveness'], spotify_song['valence'],
                spotify_song['speechiness'], spotify_song['tempo'], spotify_song['loudness'], spotify_song['key']]
    })

    # Calculate difference
    comparison['Difference'] = comparison['Your Algorithm'] - comparison['Spotify']

    # Add key error score as a separate row
    comparison = pd.concat([
        comparison,
        pd.DataFrame([{
            'Feature': 'key_error_score',
            'Your Algorithm': key_error_score,
            'Spotify': None,
            'Difference': None
        }])
    ], ignore_index=True)

    # Print the table
    logging.info(f"Comparison for: {spotify_song['name']} by {spotify_song['artist']}")
    logging.info(comparison.to_string(index=False))
    output_path = COMPARISONS_FOLDER / base_filename

    try:
        # Ensure the comparisons directory exists
        COMPARISONS_FOLDER.mkdir(parents=True, exist_ok=True) 
        # Save to the comparisons folder
        comparison.to_csv(output_path, index=False)
        logging.info(f"Comparison saved to: {output_path}")
    except OSError as e:
        # Handle potential errors during directory creation or file writing
        logging.error(f"Error creating directory or saving file {output_path}: {e}")
    except Exception as e:
        # Handle other unexpected errors during save
        logging.error(f"An unexpected error occurred while saving comparison CSV {output_path}: {e}")

# --- Core Classes ---
class SoundCloudScraper:
    """Handles searching SoundCloud and parsing results using Selenium."""

    def __init__(self):
        self.driver = None
        self.session = requests.Session() # Keep requests for potential future use
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

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

    def download_track(self, url, expected_artist, expected_title):
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

class SpotifyFeaturesClone:
    def __init__(self):
        """Initialize the audio analysis pipeline."""
        self.sample_rate = 22050  # Librosa's default sample rate
        warnings.filterwarnings('ignore')  # Suppress librosa warnings
        
    def load_audio(self, file_path):
        """Load and prepare audio file for analysis."""
        # Load with a small duration for fast feature estimation if needed
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        return y
    
    def extract_features(self, audio):
        """Extract Spotify-like audio features from the loaded audio."""
        features = {}
        
        # === Rhythm features ===
        # Tempo and beat tracking with multi-band onset detection for more accuracy
        # Use harmonic/percussive source separation to improve beat detection
        y_harmonic, y_percussive = librosa.effects.hpss(audio)
        
        # Get onset envelope from percussive component for better beat detection
        onset_env = librosa.onset.onset_strength(
            y=y_percussive, 
            sr=self.sample_rate,
            aggregate=np.median  # More robust aggregation
        )
        
        # Enhanced tempo detection with fallback
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, 
            sr=self.sample_rate,
            start_bpm=120,
            tightness=100
        )

        # Clamp to a musically realistic range
        if tempo < 60 or tempo > 180:
            # Fallback to global tempo estimation
            fallback = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sample_rate)
            if len(fallback) > 0:
                tempo = fallback[0]

        # Final clamping just to be sure
        tempo = max(60.0, min(180.0, tempo))
        features['tempo'] = float(tempo)
        
        # Improved danceability based on:
        # 1. Rhythm regularity (consistency of beat intervals)
        # 2. Low-frequency energy (bass presence)
        # 3. Pulse clarity
        
        # Get beat times and calculate regularity
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            # Higher value = more consistent beat (important for danceability)
            beat_regularity = 1.0 - min(1.0, np.std(beat_intervals) / np.mean(beat_intervals))
        else:
            beat_regularity = 0.0
            
        # Measure bass energy (important for dance music)
        spec = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        bass_mask = freqs <= 250  # Bass frequencies
        bass_energy = np.mean(spec[bass_mask]) / (np.mean(spec) + 1e-8)
        
        # Pulse clarity from onset strength
        pulse_clarity = librosa.feature.rms(y=y_percussive)[0].mean()
        
        # Combine factors with appropriate weights
        danceability_raw = (0.4 * beat_regularity + 
                           0.3 * bass_energy +
                           0.3 * self._normalize(pulse_clarity, 0, 0.2, 0, 1))
        features['danceability'] = min(1.0, max(0.0, danceability_raw))
        
        # === Energy-related features ===
        # Improved energy measure considering:
        # 1. RMS energy
        # 2. Spectral entropy (higher = more "busy" signal)
        # 3. Dynamic range
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        
        # Spectral entropy - measure of "chaos" in the signal
        stft = np.abs(librosa.stft(audio))
        stft_normalized = stft / (np.sum(stft, axis=0, keepdims=True) + 1e-8)
        spectral_entropy = -np.sum(stft_normalized * np.log2(stft_normalized + 1e-8), axis=0).mean()
        spectral_entropy_norm = self._normalize(spectral_entropy, 0, 5, 0, 1)
        
        # Dynamic range compression measure
        dynamic_range = np.percentile(rms, 95) / (np.percentile(rms, 10) + 1e-8)
        dynamic_range_norm = self._normalize(dynamic_range, 1, 20, 0, 1)
        
        # Combined energy feature
        energy_raw = 0.6 * self._normalize(np.mean(rms), 0, 0.2, 0, 1) + \
                    0.2 * spectral_entropy_norm + \
                    0.2 * dynamic_range_norm
        features['energy'] = min(1.0, max(0.0, energy_raw))
        
        # === Timbre & Spectral features ===
        # Improved spectral features
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        flatness = librosa.feature.spectral_flatness(y=audio)
        
        # Acousticness (improved): acoustic tracks have:
        # 1. Lower spectral centroid
        # 2. Lower spectral entropy
        # 3. Higher contrast between bands (less flat spectrum)
        # 4. Less percussive content
        
        # Ratio of harmonic to percussive energy
        harmonic_energy = np.mean(librosa.feature.rms(y=y_harmonic)[0])
        percussive_energy = np.mean(librosa.feature.rms(y=y_percussive)[0])
        harmonic_ratio = harmonic_energy / (percussive_energy + harmonic_energy + 1e-8)
        
        # Spectral flatness (electronic music tends to have flatter spectra)
        flatness_mean = np.mean(flatness)
        
        # Contrast between low and high frequency bands
        if contrast.shape[0] >= 6:  # Ensure we have enough contrast bands
            high_freq_energy = np.mean(contrast[-2:])  # Higher frequency bands
            low_freq_energy = np.mean(contrast[:2])    # Lower frequency bands
            contrast_ratio = low_freq_energy / (high_freq_energy + 1e-8)
        else:
            contrast_ratio = 0.5
        
        # Combine into acousticness score
        acousticness_raw = 0.4 * harmonic_ratio + \
                          0.3 * (1.0 - self._normalize(np.mean(centroid), 500, 3000, 0, 1)) + \
                          0.2 * (1.0 - flatness_mean) + \
                          0.1 * self._normalize(contrast_ratio, 0.5, 5, 0, 1)
                          
        features['acousticness'] = min(1.0, max(0.0, acousticness_raw))
        
        # === Key and mode detection (improved) ===
        # Using more robust key detection with CREMA and chroma features
        chroma_cqt = librosa.feature.chroma_cqt(
            y=y_harmonic,  # Use harmonic component for better key detection
            sr=self.sample_rate, 
            bins_per_octave=36,  # Higher resolution
            n_chroma=12
        )
        
        # Smooth the chroma to get a better key estimate
        chroma_smooth = np.minimum(1.0, librosa.decompose.nn_filter(
            chroma_cqt,
            aggregate=np.median,
            metric='cosine'
        ))
        
        # Sum over time to get key profile
        chroma_sum = np.sum(chroma_smooth, axis=1)
        
        # Key profiles for major and minor keys (Krumhansl-Kessler profiles)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalize profiles
        major_profile = major_profile / major_profile.sum()
        minor_profile = minor_profile / minor_profile.sum()
        chroma_sum = chroma_sum / chroma_sum.sum()
        
        # Calculate correlation manually to avoid np.corrcoef issues
        def manual_correlation(a, b):
            # Make sure inputs are numpy arrays
            a = np.array(a, dtype=float)
            b = np.array(b, dtype=float)
            
            # Calculate mean of each array
            a_mean = np.mean(a)
            b_mean = np.mean(b)
            
            # Calculate numerator (covariance)
            numerator = np.sum((a - a_mean) * (b - b_mean))
            
            # Calculate denominator (product of standard deviations)
            a_std = np.sqrt(np.sum((a - a_mean) ** 2))
            b_std = np.sqrt(np.sum((b - b_mean) ** 2))
            denominator = a_std * b_std
            
            # Handle division by zero
            if denominator == 0:
                return 0
            
            # Return Pearson correlation coefficient
            return numerator / denominator
        
        # Calculate correlation for all possible keys
        correlations_major = np.zeros(12)
        correlations_minor = np.zeros(12)
        
        for i in range(12):
            rolled_major = np.roll(major_profile, i)
            rolled_minor = np.roll(minor_profile, i)
            
            # Calculate correlations using our manual function
            correlations_major[i] = manual_correlation(chroma_sum, rolled_major)
            correlations_minor[i] = manual_correlation(chroma_sum, rolled_minor)
        
        # Find the best key and mode
        key_major = np.argmax(correlations_major)
        key_minor = np.argmax(correlations_minor)
        
        if np.max(correlations_major) > np.max(correlations_minor):
            key = int(key_major)
            mode = 1  # Major
            key_confidence = float(np.max(correlations_major))
        else:
            key = int(key_minor)
            mode = 0  # Minor
            key_confidence = float(np.max(correlations_minor))
        
        features['key'] = key
        features['mode'] = mode
        features['key_confidence'] = key_confidence
        
        # === Valence (musical positiveness) - improved ===
        # Factors that contribute to valence:
        # 1. Mode (major/minor)
        # 2. Tempo
        # 3. Energy
        # 4. Spectral characteristics (brightness)
        
        # Mode factor
        mode_factor = 0.7 if mode == 1 else 0.3
        
        # Tempo factor - higher tempos usually indicate higher valence
        tempo_factor = self._normalize(tempo, 60, 180, 0, 1)
        
        # Timbral brightness - brighter sounds often indicate higher valence
        brightness = self._normalize(np.mean(centroid), 500, 3000, 0, 1)
        
        # Rhythm strength - stronger rhythm often means higher valence
        rhythm_strength = self._normalize(np.mean(onset_env), 0, 0.5, 0, 1)
        
        # Combine factors for valence
        valence_raw = 0.3 * mode_factor + \
                     0.2 * features['energy'] + \
                     0.2 * tempo_factor + \
                     0.15 * brightness + \
                     0.15 * rhythm_strength
                     
        features['valence'] = min(1.0, max(0.0, valence_raw))
        
        # === Loudness (improved) ===
        # Calculate perceived loudness using multiple bands and considering perceptual weighting
        
        # More accurate loudness calculation with A-weighting (to match human perception)
        # First convert RMS to dB
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)
        
        # Apply perceptual weighting - mid frequencies contribute more to perceived loudness
        spec_band = librosa.stft(audio)
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        
        # Simplified A-weighting
        a_weighting = np.zeros_like(freqs)
        for i, f in enumerate(freqs):
            # Approximate A-weighting curve
            if f < 20:
                a_weighting[i] = -70
            elif f < 100:
                a_weighting[i] = -20 * np.log10(100 / f)
            elif f < 1000:
                a_weighting[i] = 0
            elif f < 10000:
                a_weighting[i] = 2
            else:
                a_weighting[i] = -20 * np.log10(f / 10000)
        
        # Apply weighting to spectrogram
        weighted_spec = np.abs(spec_band) * np.reshape(10**(a_weighting/20), (-1, 1))
        
        # Compute weighted loudness
        weighted_loudness = librosa.amplitude_to_db(np.mean(np.mean(weighted_spec, axis=1)))
        features['loudness'] = float(max(-60, min(0, weighted_loudness)))
        
        # === Instrumentalness (improved) ===
        # Better vocal detection using multiple factors:
        # 1. MFCCs (vocal range)
        # 2. Spectral flatness (vocals tend to have less flat spectrum)
        # 3. Pitch variation (vocals have more pitch variation)
        
        # Extract MFCCs for vocal detection
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20)
        
        # Focus on MFCCs associated with vocals (typically 2-8)
        vocal_mfccs = mfccs[2:8, :]
        
        # Vocals typically have higher variance in these MFCCs
        vocal_var = np.var(vocal_mfccs, axis=1).mean()
        
        # Pitch variation - higher for vocals
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        pitch_variation = 0
        if magnitudes.max() > 0:  # Only if we detect pitches
            # For each frame, find the highest magnitude pitch
            pitch_max_indices = np.argmax(magnitudes, axis=0)
            pitches_max = np.array([pitches[pitch_max_indices[i], i] for i in range(magnitudes.shape[1])])
            pitches_max = pitches_max[pitches_max > 0]  # Only consider frames with detected pitch
            if len(pitches_max) > 0:
                # Calculate variation in the detected pitches
                pitch_variation = np.std(pitches_max) / (np.mean(pitches_max) + 1e-8)
        
        # Combine factors - higher values mean less vocal content
        instrumental_raw = 1.0 - (0.6 * self._normalize(vocal_var, 0.1, 5, 0, 1) + 
                               0.4 * self._normalize(pitch_variation, 0, 0.5, 0, 1))
        
        features['instrumentalness'] = min(1.0, max(0.0, instrumental_raw))
        
        # === Speechiness (improved) ===
        # Better speech detection using:
        # 1. Zero crossing rate (higher for speech)
        # 2. Rhythm regularity (lower for speech)
        # 3. Spectral shape (speech has specific formant patterns)
        
        # Zero crossing rate (speech has higher values)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        
        # Speech has specific rhythm patterns - less regular than music
        if len(beat_times) > 1:
            rhythm_regularity = 1.0 - beat_regularity  # Invert - speech has less regular rhythm
        else:
            rhythm_regularity = 0.5  # Default mid-value
            
        # Spectral shape for speech detection
        # Speech tends to have specific spectral patterns due to formants
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        rolloff_mean = np.mean(spectral_rolloff)
        
        # MFCC pattern correlation with speech
        # Mean and variance of MFCCs are often used to detect speech
        mfcc_deltas = librosa.feature.delta(mfccs)
        mfcc_var = np.var(mfcc_deltas, axis=1).mean()
        
        # Combine factors for speechiness
        speechiness_raw = 0.4 * self._normalize(zcr_mean, 0.05, 0.15, 0, 1) + \
                         0.3 * rhythm_regularity + \
                         0.3 * self._normalize(mfcc_var, 0.5, 5, 0, 1)
                         
        features['speechiness'] = min(1.0, max(0.0, speechiness_raw))
        
        # === Liveness (improved) ===
        # Better live detection using:
        # 1. Audio dynamics (live recordings have more dynamic variation)
        # 2. Spectral features (crowd noise, room acoustics)
        # 3. Reverb estimation
        
        # Dynamic range in live recordings
        percentile_diff = np.percentile(rms, 95) - np.percentile(rms, 10)
        dynamic_range = self._normalize(percentile_diff, 0.01, 0.1, 0, 1)
        
        # Spectral shape for audience noise detection
        # Live recordings often have more energy in higher frequencies (applause, crowd)
        rolloff_high = np.percentile(spectral_rolloff, 90)
        high_freq_content = self._normalize(rolloff_high, 3000, 8000, 0, 1)
        
        # Reverb estimation - live recordings typically have more reverb
        # Use decay time from impulse response
        y_harmonic = librosa.effects.harmonic(audio)
        decay_envelope = librosa.onset.onset_strength(y=y_harmonic, sr=self.sample_rate)
        decay_time = 0
        if len(decay_envelope) > 1:
            # Estimate decay time using envelope
            decay_segments = librosa.util.frame(decay_envelope, frame_length=10, hop_length=1)
            if decay_segments.shape[1] > 0:
                segment_means = np.mean(decay_segments, axis=0)
                if len(segment_means) > 1:
                    # Calculate slope of decay
                    indices = np.arange(len(segment_means))
                    slope, _, _, _, _ = stats.linregress(indices, segment_means)
                    # Negative slope indicates decay - steeper is less reverb
                    decay_time = self._normalize(abs(slope), 0.001, 0.1, 0, 1)
                    # Invert so higher value = more reverb
                    decay_time = 1 - decay_time
        
        # Combine factors for liveness score
        liveness_raw = 0.4 * dynamic_range + \
                      0.3 * high_freq_content + \
                      0.3 * decay_time
                      
        features['liveness'] = min(1.0, max(0.0, liveness_raw))
        
        return features
    
    def _normalize(self, value, min_val, max_val, new_min, new_max):
        """Normalize a value to a new range."""
        if value < min_val:
            value = min_val
        if value > max_val:
            value = max_val
            
        normalized = (value - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
        return float(normalized)
    
    def _convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types for JSON serialization."""
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def analyze_track(self, file_path):
        """Analyze a track and return Spotify-like audio features."""
        try:
            audio = self.load_audio(file_path)
            features = self.extract_features(audio)
            return features
        except Exception as e:
            logging.error(f"Error analyzing track: {e}")
            return None

    def save_features_to_cache(self, file_path, features_dict):
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

    def get_features_from_cache(self, file_path):
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
        self.analyzer = SpotifyFeaturesClone()
        self.downloaded_songs_paths = []
        # Ensure download folder exists
        self.download_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using download folder: {self.download_folder.resolve()}")
        logging.info(f"Using checkpoint file: {self.checkpoint_file.resolve()}")

    
    def _load_checkpoint(self):
        """Loads checkpoint data from the JSON file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
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
        results = df[['name', 'artist']][start_index:end_index]
        results = results.drop_duplicates()
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
            self.downloaded_songs_paths.append(DOWNLOAD_FOLDER + Path(output_file).name)
            return

        # if current_status == 'failed_ytdlp':
        #      logging.warning(f"Checkpoint indicates previous yt-dlp download failed for {soundcloud_url}. Skipping.")
        #      # Optionally add logic here to retry failed downloads after a certain condition
        #      return
        
        logging.info(f"Attempting download for {soundcloud_url}...")
        
        # 4. Download using YTDLPDownloader
        # Pass original names for filename template, downloader might refine based on metadata
        final_filename, download_successful = self.downloader.download_track(
            soundcloud_url, 
            artist_name, 
            song_name 
        )

        # 5. Update Checkpoint based on download result
        if download_successful and final_filename:
            existing_entry['download_status'] = 'completed'
            existing_entry['output_file'] = Path(final_filename).name # Store just the filename
            self.downloaded_songs_paths.append(DOWNLOAD_FOLDER + Path(final_filename).name)

        else:
            existing_entry['download_status'] = 'failed_ytdlp'
            existing_entry['output_file'] = None # Ensure no output file recorded on failure
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

                self.process_song(song_name, artist_name)
                total_processed += 1
                logging.info(f"Completed {total_processed}/{len(self.song_list)}")

        finally:
            # Ensure Selenium driver for scraper is closed when pipeline finishes or errors out
            logging.info("Pipeline run finished. Cleaning up SoundCloudScraper driver...")
            self.scraper._quit_driver()
            logging.info("Cleanup complete.")

    def analyze_songs(self):
        """Analyzes downloaded songs and compares them to Spotify features."""
        
        if not self.downloaded_songs_paths:
            self.downloaded_songs_paths = [f"{DOWNLOAD_FOLDER}/{f}" for f in os.listdir(DOWNLOAD_FOLDER) if os.path.isfile(os.path.join(DOWNLOAD_FOLDER, f))]

        total_songs = len(self.downloaded_songs_paths)
        i=1

        for file_path in self.downloaded_songs_paths:
            
            logging.info(f"Analyzing file: {file_path}")
            if file_path != "failed":
                if os.path.exists(file_path) and file_path.endswith(".mp3"):
                    logging.info(f"the file path is: {file_path}")
                    features = self.analyzer.get_features_from_cache(file_path)
                    if not features:
                        results = self.analyzer.analyze_track(file_path)
                        # Convert NumPy types to Python native types before JSON serialization
                        json_safe_results = self.analyzer._convert_numpy_types(results)
                        self.analyzer.save_features_to_cache(file_path, json_safe_results)
                        features = self.analyzer.get_features_from_cache(file_path)
                    
                    # Print results in the specified order
                    ordered_features = [
                        "danceability", "energy", "key", "loudness", "mode", 
                        "speechiness", "acousticness", "instrumentalness", 
                        "liveness", "valence", "tempo"
                    ]

                    name_artist = file_path.split("/")[-1]
                    print(f"this is name_artist: {name_artist}")
                    artist, song = name_artist[:-4].split(" - ",1)

                    print(f"artist: {artist} and title: {song} .")
                    # time.sleep(5)
                    ordered_results = {feature: features.get(feature, "N/A") for feature in ordered_features}
                    logging.info("we got the results from analyzer, comparing to spotify now")
                    compare_results(ordered_results, song, artist)
                    logging.info(f"we got the results from compare_results, {i}/{total_songs} done. Printing the results")
                    # logging.info(json.dumps(ordered_results, indent=2))
                    i+=1
                else:
                    logging.info(f"File not found: {file_path}")
        


    
