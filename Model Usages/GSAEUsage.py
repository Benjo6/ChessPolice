import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
import os

# ====== HARDCODED PATHS ======
MODEL_PATH = './GSAE/Bullet2300/GSAE_model_2300Bullet'
PREPROCESSOR_PATH = './GSAE/Bullet2300/GSAE_preprocessor_2300Bullet.pkl'
TEST_DATA_PATH = './Evaluation/Bullet/2300/cheating_GSAE_2300.csv'
OUTPUT_PATH = './Results/Bullet/2300/GSAE/cheating_GSAE_2300Res.csv'

class AnomalyDetector:
    def __init__(self):
        # Load trained model and preprocessor
        self.model = tf.keras.models.load_model(MODEL_PATH)
        with open(PREPROCESSOR_PATH, 'rb') as f:
            pp = pickle.load(f)
            self.scaler = pp['scaler']
            self.threshold = pp['threshold']
            self.feature_names = pp['features']
            self.config = pp['config']

    def _preprocess(self, data):
        """Match training data preprocessing exactly"""
        # Validate required features
        missing_features = set(self.feature_names) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and clean features
        processed = data[self.feature_names].copy()
        
        # Convert to numeric, handle infinities
        processed = processed.apply(pd.to_numeric, errors='coerce')
        processed = processed.replace([np.inf, -np.inf], np.nan)
        
        # Apply same rounding as training
        processed = processed.round(self.config['max_decimal'])
        
        # Drop rows with missing values
        processed = processed.dropna()
        
        return processed

    def detect_anomalies(self, data_path):
        # Load and preprocess data
        raw_data = pd.read_csv(data_path)
        game_ids = raw_data['Game ID'] if 'Game ID' in raw_data else None
        processed = self._preprocess(raw_data)
        
        if processed.empty:
            raise ValueError("No valid data after preprocessing")
        
        # Scale using training scaler
        scaled_data = self.scaler.transform(processed)
        
        # Get reconstructions
        reconstructions = self.model.predict(scaled_data, verbose=0)
        
        # Calculate reconstruction errors
        errors = np.mean(np.abs(scaled_data - reconstructions), axis=1)
        
        # Generate results
        results = pd.DataFrame({
            'reconstruction_error': errors,
            'is_anomaly': errors > self.threshold,
            'threshold': self.threshold
        })
        
        # Add game IDs if available
        if game_ids is not None:
            results['Game ID'] = game_ids.iloc[processed.index].values
        
        # Identify most suspicious features
        feature_errors = np.abs(scaled_data - reconstructions)
        results['suspicious_features'] = [
            self.feature_names[np.argmax(row)] if is_anom else ''
            for row, is_anom in zip(feature_errors, results['is_anomaly'])
        ]
        
        return results

def main():
    detector = AnomalyDetector()
    results = detector.detect_anomalies(TEST_DATA_PATH)
    
    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    
    # Print summary
    total = len(results)
    anomalies = results['is_anomaly'].sum()
    print(f"Analyzed {total} games")
    print(f"Detected anomalies: {anomalies} ({anomalies/total:.1%})")
    print(f"Threshold: {results['threshold'].iloc[0]:.4f}")
    print(f"Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()