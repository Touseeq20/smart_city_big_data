import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import joblib

RAW_DATA_PATH = os.path.join("data", "raw", "Metro_Interstate_Traffic_Volume.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed")
PROCESSED_FILE = os.path.join(PROCESSED_DATA_PATH, "traffic_processed.csv")
LABEL_ENCODER_PATH = os.path.join("data", "processed", "weather_encoder.pkl")

def load_and_preprocess():
    """Loads raw data, cleans it, performs feature engineering, and saves processed data."""
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data not found at {RAW_DATA_PATH}. Run data_loader.py first.")
        return None

    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 1. Date Conversion
    print("Converting dates...")
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # 2. Sorting
    df = df.sort_values('date_time').reset_index(drop=True)
    
    # 3. Feature Engineering
    print("Feature engineering...")
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek # 0=Monday, 6=Sunday
    df['month'] = df['date_time'].dt.month
    df['year'] = df['date_time'].dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Categorize hours into time slots
    def get_time_slot(h):
        if 6 <= h < 10: return 'Morning Rush'
        elif 10 <= h < 16: return 'Work Hours'
        elif 16 <= h < 20: return 'Evening Rush'
        else: return 'Off Peak'
    
    df['time_slot'] = df['hour'].apply(get_time_slot)
    
    # 4. Encoding Categorical Data
    print("Encoding categorical features...")
    le_main = LabelEncoder()
    df['weather_main_code'] = le_main.fit_transform(df['weather_main'])
    
    le_desc = LabelEncoder()
    df['weather_description_code'] = le_desc.fit_transform(df['weather_description'])
    
    # Save Encoders for future inference if needed
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
        
    joblib.dump({'main': le_main, 'desc': le_desc}, LABEL_ENCODER_PATH)
    
    # 5. Handling Duplicates (if any)
    df = df.drop_duplicates(subset=['date_time'], keep='first')
    
    # 6. Save Processed Data
    print(f"Saving processed data to {PROCESSED_FILE}...")
    df.to_csv(PROCESSED_FILE, index=False)
    print("Preprocessing complete.")
    return df

if __name__ == "__main__":
    load_and_preprocess()
