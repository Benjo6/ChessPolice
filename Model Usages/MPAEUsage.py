import tensorflow as tf
import pandas as pd
import numpy as np
import os

# ======== CONFIGURATION ========
MODEL_DIR = "./MPAE/Bullet2300/mpae_model_2300_Bullet"
TOKENIZER_DIR = "./MPAE/Bullet2300/move_tokenizer_2300_Bullet"
INPUT_CSV = "./Evaluation/Bullet/2300/cheating_MPAE_2300.csv"
OUTPUT_CSV = "./Results/Bullet/2300/MPAE/cheating_MPAE_2300Res.csv"
MAX_MOVES = 75
# ===============================

class CheatDetector:
    def __init__(self):
        # Validate paths
        if not os.path.isdir(MODEL_DIR):
            raise FileNotFoundError(f"Model directory missing: {os.path.abspath(MODEL_DIR)}")
        if not os.path.isdir(TOKENIZER_DIR):
            raise FileNotFoundError(f"Tokenizer directory missing: {os.path.abspath(TOKENIZER_DIR)}")

        # Load model and tokenizer
        self.model = tf.keras.models.load_model(MODEL_DIR)
        self.tokenizer = tf.saved_model.load(TOKENIZER_DIR)
        self.tokenize_fn = self.tokenizer.signatures['serving_default']

        # Get normalization parameters from model
        self.feat_means = [self.model.mean_error.numpy()] * 3
        self.feat_stds = [self.model.std_error.numpy()] * 3

    def _preprocess_game(self, game_moves, game_features):
        """Process individual game data with proper tokenization"""
        # Tokenize using saved signature
        encoded = self.tokenize_fn(tf.constant([" ".join(game_moves)]))['output_0'].numpy()[0]

        # Pad and scale moves
        padded_moves = tf.keras.preprocessing.sequence.pad_sequences(
            [encoded], 
            maxlen=MAX_MOVES,
            padding='post',
            truncating='post',
            value=0
        ).astype(np.float32) / self.tokenizer.vocabulary_size.numpy() 

        # Process numerical features
        processed_features = np.zeros((MAX_MOVES, 3), dtype=np.float32)
        for idx, (name, values) in enumerate(game_features.items()):
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                [values],
                maxlen=MAX_MOVES,
                padding='post',
                truncating='post',
                dtype=np.float32
            )[0]
            processed_features[:, idx] = (padded - self.feat_means[idx]) / self.feat_stds[idx]

        # Combine features
        combined = np.zeros((1, MAX_MOVES, 4), dtype=np.float32)
        combined[0, :, 0] = padded_moves[0]
        combined[0, :, 1:] = processed_features
        
        return combined.reshape(1, -1)

    def _preprocess(self):
        """Batch preprocessing with validation"""
        df = pd.read_csv(INPUT_CSV)
        games = df.groupby('Game ID')
        
        processed_data = []
        game_ids = []
        game_groups = {}

        for game_id, group in games:
            # Validate input format
            required_columns = {'Move', 'Eval Diff', 'Time Spent', 'Board Complexity'}
            if not required_columns.issubset(group.columns):
                raise ValueError(f"Missing columns in game {game_id}")

            # Process each game
            game_features = {
                'Eval Diff': group['Eval Diff'].values,
                'Time Spent': group['Time Spent'].values,
                'Board Complexity': group['Board Complexity'].values
            }
            
            processed = self._preprocess_game(
                group['Move'].tolist(),
                game_features
            )
            processed_data.append(processed)
            game_ids.append(game_id)
            game_groups[game_id] = group.reset_index(drop=True)
            
        return np.vstack(processed_data), game_ids, game_groups

    def detect(self):
        """Run detection pipeline with move-level anomaly identification"""
        processed_data, game_ids, game_groups = self._preprocess()
        
        # Get model predictions
        reconstructions = self.model.predict(processed_data)
        game_errors = tf.reduce_mean(tf.square(processed_data - reconstructions), axis=1).numpy()
        
        # Reshape for move-level analysis (games, MAX_MOVES, 4)
        original_moves = processed_data.reshape(-1, MAX_MOVES, 4)
        reconstructed_moves = reconstructions.reshape(-1, MAX_MOVES, 4)
        
        # Move-level anomaly detection
        anomaly_details = []
        for i, game_id in enumerate(game_ids):
            group = game_groups[game_id]
            move_count = len(group)
            move_errors = np.mean(
                (original_moves[i,:move_count] - reconstructed_moves[i,:move_count])**2, 
                axis=1
            )
            # Dynamic thresholding (mean + 2*std)
            threshold = np.mean(move_errors) + 2*np.std(move_errors)
            anomalous_indices = np.where(move_errors > threshold)[0]
            # If none, take top 3
            if len(anomalous_indices) == 0 and move_count > 0:
                anomalous_indices = np.argsort(move_errors)[-3:]
            anomalous_moves = []
            for idx in anomalous_indices:
                if idx < move_count:
                    move_row = group.iloc[idx]
                    player = "White" if idx % 2 == 0 else "Black"
                    z_score = (move_errors[idx] - np.mean(move_errors)) / (np.std(move_errors) + 1e-10)
                    anomalous_moves.append({
                        'move_number': move_row['MoveNumber'] if 'MoveNumber' in move_row else idx+1,
                        'player': player,
                        'notation': move_row['Move'],
                        'error': move_errors[idx],
                        'z_score': z_score
                    })
            # Sort by descending error
            anomalous_moves.sort(key=lambda x: x['error'], reverse=True)
            # Format for output
            move_strings = [
                f"Move {m['move_number']} ({m['player']}): {m['notation']} (Error: {m['error']:.4f}, Z: {m['z_score']:.2f})" 
                for m in anomalous_moves
            ]
            anomaly_details.append({
                'GameID': game_id,
                'AnomalousMoves': "; ".join(move_strings) if move_strings else "No significant anomalies"
            })

        # Compile results
        z_scores = (game_errors - self.model.mean_error.numpy()) / self.model.std_error.numpy()
        roi_scores = 50 + z_scores * 10  # Adjusted scaling factor

        results = pd.DataFrame({
            'GameID': game_ids,
            'Reconstruction_Error': game_errors.flatten(),
            'Threshold': self.model.threshold.numpy(),
            'ROI_Score': roi_scores.flatten()
        })

        anomaly_df = pd.DataFrame(anomaly_details)
        results = pd.merge(results, anomaly_df, on='GameID')

        # Classify suspicion levels
        results['Suspicion_Level'] = np.select(
            [
                results['ROI_Score'] < 60,
                (results['ROI_Score'] >= 60) & (results['ROI_Score'] < 70),
                (results['ROI_Score'] >= 70) & (results['ROI_Score'] < 80),
                results['ROI_Score'] >= 80
            ],
            ['Normal', 'Amber', 'Orange', 'Red'],
            default='Unknown'
        )

        # Save results
        try:
            results.to_csv(OUTPUT_CSV, index=False)
            print(f"Results saved to {OUTPUT_CSV}")
        except Exception as e:
            raise IOError(f"Failed to save results: {str(e)}") from e

if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")
    print(f"Model exists: {os.path.isdir(MODEL_DIR)}")
    print(f"Tokenizer exists: {os.path.isdir(TOKENIZER_DIR)}")
    detector = CheatDetector()
    detector.detect()