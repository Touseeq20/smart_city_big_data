import os
import urllib.request
import gzip
import shutil
import pandas as pd

RAW_DATA_PATH = os.path.join("data", "raw")
PROCESSED_DATA_PATH = os.path.join("data", "processed")
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
COMPRESSED_FILE = os.path.join(RAW_DATA_PATH, "Metro_Interstate_Traffic_Volume.csv.gz")
CSV_FILE = os.path.join(RAW_DATA_PATH, "Metro_Interstate_Traffic_Volume.csv")

def download_data():
    """Downloads the dataset from UCI Archive if it doesn't exist."""
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
    
    if os.path.exists(CSV_FILE):
        print(f"Dataset already exists at {CSV_FILE}")
        return

    if not os.path.exists(COMPRESSED_FILE):
        print(f"Downloading dataset from {DATASET_URL}...")
        try:
            urllib.request.urlretrieve(DATASET_URL, COMPRESSED_FILE)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            return

    print("Extracting dataset...")
    try:
        with gzip.open(COMPRESSED_FILE, 'rb') as f_in:
            with open(CSV_FILE, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted to {CSV_FILE}")
    except Exception as e:
        print(f"Error extracting data: {e}")

if __name__ == "__main__":
    download_data()
