import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

PROCESSED_DATA_PATH = os.path.join("data", "processed", "traffic_processed.csv")
MODELS_DIR = os.path.join("data", "models")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

class TrafficMiner:
    def __init__(self):
        self.df = None
        self.kmeans_model = None
        self.anomaly_model = None
        self.rf_model = None

    def load_data(self):
        if os.path.exists(PROCESSED_DATA_PATH):
            self.df = pd.read_csv(PROCESSED_DATA_PATH)
            return True
        return False

    def train_clustering(self, n_clusters=4):
        """Groups traffic patterns into clusters."""
        if self.df is None: return
        
        # Features for clustering: Traffic Vol, Hour, Day of Week
        features = self.df[['traffic_volume', 'hour', 'day_of_week']]
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = self.kmeans_model.fit_predict(features)
        
        # Save model
        joblib.dump(self.kmeans_model, os.path.join(MODELS_DIR, "kmeans_traffic.pkl"))
        return self.df

    def detect_anomalies(self, contamination=0.01):
        """Detects anomalous traffic volumes."""
        if self.df is None: return
        
        features = self.df[['traffic_volume', 'hour']]
        
        self.anomaly_model = IsolationForest(contamination=contamination, random_state=42)
        self.df['anomaly'] = self.anomaly_model.fit_predict(features)
        # -1 is anomaly, 1 is normal. Map to boolean for easier usage.
        self.df['is_anomaly'] = self.df['anomaly'] == -1
        
        joblib.dump(self.anomaly_model, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
        return self.df

    def train_prediction_model(self):
        """Trains a Random Forest Regressor to predict traffic volume."""
        if self.df is None: return
        
        feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'weather_main_code', 'weather_description_code', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']
        target_col = 'traffic_volume'
        
        X = self.df[feature_cols]
        y = self.df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Use all cores
        self.rf_model.fit(X_train, y_train)
        
        y_pred = self.rf_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {"MAE": mae, "R2": r2}
        joblib.dump(self.rf_model, os.path.join(MODELS_DIR, "rf_traffic_predictor.pkl"))
        print(f"Model Trained. MAE: {mae:.2f}, R2: {r2:.2f}")
        return metrics

if __name__ == "__main__":
    miner = TrafficMiner()
    if miner.load_data():
        print("Training Clustering...")
        miner.train_clustering()
        print("Detecting Anomalies...")
        miner.detect_anomalies()
        print("Training Prediction Model...")
        miner.train_prediction_model()
        
        # Save enriched data
        miner.df.to_csv(os.path.join("data", "processed", "traffic_enhanced.csv"), index=False)
        print("Mining complete. Enhanced data saved.")
