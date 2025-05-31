import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, TimeDistributed, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import chess
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pickle

np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISM'] = '1'

class ChessFeatureEngineering:
    """
    Class to handle chess-specific feature engineering and preprocessing
    """
    def __init__(self, scaler_file='lstm_bullet_2275_2325_scalers.pkl'):
        self.scalers = {}
        self.scaler_file = scaler_file

    def save_scalers(self):
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(self.scalers, f)
        print(f"Scalers saved to {self.scaler_file}")

    def load_scalers(self):
            """
            Load scalers from the scaler file if it exists.
            """
            if os.path.exists(self.scaler_file):
                with open(self.scaler_file, 'rb') as f:
                    self.scalers = pickle.load(f)
                print(f"Scalers loaded from {self.scaler_file}")
                return True
            else:
                print(f"Scaler file {self.scaler_file} not found.")
                return False
            
    def extract_advanced_features(self, df):
        enhanced_df = df.copy()
        enhanced_df['Move_Quality'] = 1.0 - np.abs(enhanced_df['Eval Diff']) / 10.0
        time_by_game = enhanced_df.groupby(['Game ID', enhanced_df['Move No.'].apply(lambda x: x % 2 == 1)])['Time Spent'].mean()
        time_by_game = time_by_game.reset_index()
        time_by_game.columns = ['Game ID', 'IsWhite', 'Avg_Time']
        enhanced_df['IsWhite'] = (enhanced_df['Move No.'] % 2 == 1).astype(int)
        enhanced_df = pd.merge(enhanced_df, time_by_game, on=['Game ID', 'IsWhite'], how='left')
        enhanced_df['Time_Ratio'] = enhanced_df['Time Spent'] / (enhanced_df['Avg_Time'] + 1e-5)
        enhanced_df['Complex_Performance'] = enhanced_df['Move_Quality'] * enhanced_df['Board Complexity']
        enhanced_df['Critical_Performance'] = enhanced_df['Move_Quality'] * enhanced_df['Volatility Score']
        enhanced_df['Engine_Alignment'] = enhanced_df['Move Match (MM)'] + enhanced_df['Equal-Value (EV)'] * 0.8
        enhanced_df['Engine_Score'] = np.exp(-np.abs(enhanced_df['Eval Diff']))
        enhanced_df['Non_Book'] = 1 - enhanced_df['Book Move']
        game_groups = enhanced_df.groupby(['Game ID', 'IsWhite'])
        
        def add_rolling_features(group):
            if len(group) > 5:
                group['Rolling_Quality_Mean'] = group['Move_Quality'].rolling(5, min_periods=3).mean()
                group['Rolling_Quality_Std'] = group['Move_Quality'].rolling(5, min_periods=3).std().fillna(0)
                group['Consistency_Score'] = 1 / (group['Rolling_Quality_Std'] + 0.1)
            else:
                group['Rolling_Quality_Mean'] = group['Move_Quality']
                group['Rolling_Quality_Std'] = 0
                group['Consistency_Score'] = 1
            return group
        
        enhanced_df = pd.concat([add_rolling_features(group) for _, group in game_groups])
        enhanced_df['Thinking_Ratio'] = enhanced_df['Time Spent'] / (enhanced_df['Board Complexity'] + 0.1)
        enhanced_df['Human_Inaccuracy_Score'] = (
            enhanced_df['Non_Book'] * 
            (1 - enhanced_df['Engine_Alignment']) * 
            (enhanced_df['Board Complexity'] + 0.2) * 
            (enhanced_df['Volatility Score'] + 0.2)
        )
        
        def extract_fen_features(fen):
            try:
                board = chess.Board(fen)
                piece_count = len(board.piece_map())
                pawn_count = sum(1 for p in board.piece_map().values() if p.piece_type == chess.PAWN)
                attack_count = sum(1 for sq in chess.SQUARES if board.is_attacked_by(chess.WHITE, sq) or board.is_attacked_by(chess.BLACK, sq))
                attacked_pieces = sum(1 for sq, piece in board.piece_map().items() 
                                    if board.is_attacked_by(chess.WHITE if piece.color == chess.BLACK else chess.BLACK, sq))
                king_safety_w = len(list(board.attackers(chess.BLACK, board.king(chess.WHITE)))) if board.king(chess.WHITE) else 0
                king_safety_b = len(list(board.attackers(chess.WHITE, board.king(chess.BLACK)))) if board.king(chess.BLACK) else 0
                return {
                    'piece_density': piece_count / 32,
                    'pawn_structure': pawn_count / 16,
                    'position_tension': attack_count / 64,
                    'piece_pressure': attacked_pieces / (piece_count + 1e-5),
                    'king_danger': (king_safety_w + king_safety_b) / 2
                }
            except Exception:
                return {
                    'piece_density': 0.5,
                    'pawn_structure': 0.5,
                    'position_tension': 0.5,
                    'piece_pressure': 0,
                    'king_danger': 0
                }
        
        fen_features = enhanced_df['FEN Before'].apply(extract_fen_features).apply(pd.Series)
        enhanced_df = pd.concat([enhanced_df, fen_features], axis=1)
        enhanced_df = enhanced_df.fillna(0)
        return enhanced_df
    
    def normalize_features(self, df, train=True , force_fit_scalers=False):
        std_features = ['Eval Diff', 'Time Spent', 'Time_Ratio', 'Board Complexity', 
                       'Volatility Score', 'Complex_Performance', 'Critical_Performance',
                       'Rolling_Quality_Mean', 'Rolling_Quality_Std', 'Consistency_Score',
                       'Human_Inaccuracy_Score', 'piece_density', 'position_tension', 
                       'piece_pressure', 'king_danger']
        minmax_features = ['Move_Quality', 'Engine_Score', 'Engine_Alignment']

        # Try to load existing scalers
        scalers_loaded = self.load_scalers() if os.path.exists(self.scaler_file) else False


        if train and (force_fit_scalers or not scalers_loaded):            
            for feature in std_features:
                if feature in df.columns:
                    self.scalers[feature] = StandardScaler()
                    df[feature] = self.scalers[feature].fit_transform(df[[feature]].values.reshape(-1, 1))
            for feature in minmax_features:
                if feature in df.columns:
                    self.scalers[feature] = MinMaxScaler()
                    df[feature] = self.scalers[feature].fit_transform(df[[feature]].values.reshape(-1, 1))
            if force_fit_scalers:
                self.save_scalers()
        else:
            if not self.scalers:
                self.load_scalers()
            for feature in std_features + minmax_features:
                            if feature in df.columns:
                                if feature not in self.scalers:
                                    raise ValueError(f"No scaler found for feature {feature} in {self.scaler_file}")
                                df[feature] = self.scalers[feature].transform(df[[feature]].values.reshape(-1, 1))
        return df
        
    def prepare_sequences(self, df, max_len=200):
        model_features = [
            'IsWhite', 'Book Move', 'Move Match (MM)', 'Equal-Value (EV)', 'Critical Time Binary',
            'Eval Diff', 'Time Spent', 'Board Complexity', 'Volatility Score', 
            'Move_Quality', 'Time_Ratio', 'Complex_Performance', 'Critical_Performance',
            'Engine_Score', 'Engine_Alignment', 'Non_Book', 'Rolling_Quality_Mean',
            'Consistency_Score', 'Human_Inaccuracy_Score', 'piece_density', 
            'position_tension', 'piece_pressure', 'king_danger'
        ]
        metadata_features = [
            'Move No.', 'Move', 'FEN Before', 'Material Balance', 'Remaining Time Before Move'
        ]
        game_groups = df.groupby('Game ID')
        sequences = []
        metadata = []
        for game_id, group in game_groups:
            group_sorted = group.sort_values('Move No.')
            features = []
            for feature in model_features:
                if feature in group_sorted.columns:
                    features.append(group_sorted[feature].values)
                else:
                    features.append(np.zeros(len(group_sorted)))
            feature_array = np.column_stack(features)
            seq_len = len(feature_array)
            if seq_len > max_len:
                padded_sequence = feature_array[:max_len]
                mask = np.ones(max_len)
            else:
                padded_sequence = np.zeros((max_len, len(model_features)))
                padded_sequence[:seq_len] = feature_array
                mask = np.zeros(max_len)
                mask[:seq_len] = 1
            sequences.append((padded_sequence, mask))
            meta_dict = {
                'game_id': game_id,
                'move_data': group_sorted[metadata_features].to_dict('records'),
                'sequence_length': seq_len
            }
            metadata.append(meta_dict)
        return sequences, metadata, model_features

class ChessAnomalyDetector:
    """
    LSTM-based anomaly detector for chess games
    """
    def __init__(self, input_dim, max_seq_len=200, model_path='chess_anomaly_detector.keras'):
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.model = None
        self.encoder = None
        self.baseline_stats = None
        self.feature_importance = None
        self.model_path = model_path
    
    def build_model(self):
        encoder_input = Input(shape=(self.max_seq_len, self.input_dim))
        x = Bidirectional(LSTM(64, return_sequences=True, 
                              dropout=0.2, recurrent_dropout=0.2, 
                              kernel_regularizer=l2(1e-4)))(encoder_input)
        x = LayerNormalization()(x)
        x = Bidirectional(LSTM(32, return_sequences=True, 
                              dropout=0.2, recurrent_dropout=0.2, 
                              kernel_regularizer=l2(1e-4)))(x)
        x = LayerNormalization()(x)
        encoded = Bidirectional(LSTM(16, return_sequences=True, 
                                   dropout=0.2, recurrent_dropout=0.2))(x)
        x = Bidirectional(LSTM(32, return_sequences=True, 
                              dropout=0.2, recurrent_dropout=0.2))(encoded)
        x = LayerNormalization()(x)
        x = Bidirectional(LSTM(64, return_sequences=True, 
                              dropout=0.2, recurrent_dropout=0.2))(x)
        x = LayerNormalization()(x)
        decoder_output = TimeDistributed(Dense(self.input_dim))(x)
        self.model = Model(encoder_input, decoder_output)
        self.encoder = Model(encoder_input, encoded)
        self.model.compile(optimizer='adam', loss='mse')
        return self.model
    
    def save_state(self, state_path):
        """
        Save baseline_stats and feature_importance to a file.
        """
        state = {
            'baseline_stats': self.baseline_stats,
            'feature_importance': self.feature_importance
        }
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"State saved to {state_path}")

    def load_state(self, state_path):
        """
        Load baseline_stats and feature_importance from a file.
        """
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        self.baseline_stats = state['baseline_stats']
        self.feature_importance = state['feature_importance']
        print(f"State loaded from {state_path}")
    
    def load_model(self):
        """
        Load a pre-trained model and its state from files.
        """
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
            encoder_input = self.model.input
            encoded_layer = [layer for layer in self.model.layers if 'bidirectional_2' in layer.name][0].output
            self.encoder = Model(encoder_input, encoded_layer)
            state_path = self.model_path.replace('.keras', '_state.pkl')
            if os.path.exists(state_path):
                self.load_state(state_path)
            else:
                raise FileNotFoundError(f"State file {state_path} not found.")
        else:
            raise FileNotFoundError(f"Model file {self.model_path} not found.")
    
    def train(self, sequences, epochs=50, batch_size=16, validation_split=0.2):
        if not self.model:
            self.build_model()
        X = np.array([seq for seq, _ in sequences])
        masks = np.array([mask for _, mask in sequences])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint(self.model_path, save_best_only=True)
        sample_weights = masks.reshape((masks.shape[0], masks.shape[1], 1))
        history = self.model.fit(
            X, X, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, checkpoint],
            sample_weight=sample_weights
        )
        self._compute_baseline_stats(X, masks)
        self._calculate_feature_importance(X, masks)
        state_path = self.model_path.replace('.keras', '_state.pkl')
        self.save_state(state_path)
        return history
    
    def _compute_baseline_stats(self, X, masks):
        reconstructions = self.model.predict(X)
        errors = np.square(X - reconstructions)
        masked_errors = errors * masks[..., np.newaxis]
        valid_errors = masked_errors[np.where(masks)]
        self.baseline_stats = {
            'overall': {
                'mean': np.mean(valid_errors),
                'std': np.std(valid_errors)
            }
        }
    
    def _calculate_feature_importance(self, X, masks):
        reconstructions = self.model.predict(X)
        errors = np.abs(X - reconstructions)
        masked_errors = errors * masks[..., np.newaxis]
        feature_errors = np.mean(masked_errors, axis=(0, 1))
        self.feature_importance = feature_errors / np.sum(feature_errors)

    def detect_anomalies(self, sequence, mask, feature_names=None):
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)
            mask = np.expand_dims(mask, axis=0)
        reconstruction = self.model.predict(sequence)
        errors = np.square(sequence - reconstruction)
        if self.feature_importance is not None:
            weighted_errors = np.average(errors[0], axis=1, weights=self.feature_importance)
        else:
            weighted_errors = np.mean(errors[0], axis=1)
        valid_indices = np.where(mask[0].astype(bool))[0]
        valid_errors = weighted_errors[valid_indices]
        overall_mean = self.baseline_stats['overall']['mean']
        overall_std = self.baseline_stats['overall']['std']
        z_scores = (valid_errors - overall_mean) / overall_std
        seq_len = len(valid_errors)
        adaptive_threshold = 3.0 if seq_len > 30 else (4.0 if seq_len > 15 else 5.0)
        suspicious_indices = np.where(z_scores > adaptive_threshold)[0]
        suspicious_moves = valid_indices[suspicious_indices]
        is_white = sequence[0, valid_indices, 0]
        white_indices = np.where(is_white == 1)[0]
        black_indices = np.where(is_white == 0)[0]
        white_z_scores = z_scores[white_indices] if len(white_indices) > 0 else []
        black_z_scores = z_scores[black_indices] if len(black_indices) > 0 else []
        white_cheat_prob = np.mean(white_z_scores > adaptive_threshold) if len(white_z_scores) > 0 else 0
        black_cheat_prob = np.mean(black_z_scores > adaptive_threshold) if len(black_z_scores) > 0 else 0
        result = {
            'white_cheat_prob': white_cheat_prob,
            'black_cheat_prob': black_cheat_prob,
            'suspicious_moves': suspicious_moves,
            'move_z_scores': dict(zip(valid_indices, z_scores)),
            'move_errors': dict(zip(valid_indices, valid_errors)),
            'adaptive_threshold': adaptive_threshold,
            'white_avg_z': np.mean(white_z_scores) if len(white_z_scores) > 0 else 0,
            'black_avg_z': np.mean(black_z_scores) if len(black_z_scores) > 0 else 0,
            'perfection_score': 1 - np.mean(z_scores > 0) if len(z_scores) > 0 else 1
        }
        return result

class ChessCheatingDetector:
    """
    Main class for chess cheating detection system
    """
    def __init__(self, max_seq_len=200, scaler_file='Models/Scaling/lstm_blitz_2275_2325_scalers.pkl', model_path='Models/MSLSTM_detector_blitz_2275_2325.keras'):
        self.feature_engineering = ChessFeatureEngineering(scaler_file=scaler_file)
        self.anomaly_detector = None
        self.max_seq_len = max_seq_len
        self.feature_names = None
        self.scaler_file = scaler_file
        self.model_path = model_path
    
    def prepare_data(self, data_path, train=True, force_fit_scalers=False):
        print(data_path)
        df = pd.read_csv(data_path)
        enhanced_df = self.feature_engineering.extract_advanced_features(df)
        normalized_df = self.feature_engineering.normalize_features(enhanced_df, train=train, force_fit_scalers=force_fit_scalers)
        sequences, metadata, feature_names = self.feature_engineering.prepare_sequences(
            normalized_df, max_len=self.max_seq_len
        )
        self.feature_names = feature_names
        return sequences, metadata
    
    def train_model(self, sequences, epochs=50, batch_size=16):
        input_dim = sequences[0][0].shape[1]
        self.anomaly_detector = ChessAnomalyDetector(input_dim, self.max_seq_len, self.model_path)
        self.anomaly_detector.build_model()
        history = self.anomaly_detector.train(sequences, epochs, batch_size)
        return history
    
    def load_model(self):
        if not self.anomaly_detector:
            input_dim = None
            self.anomaly_detector = ChessAnomalyDetector(input_dim, self.max_seq_len, self.model_path)
        self.anomaly_detector.load_model()
    
    def save_model(self):
        """
        Save the trained model and its state
        """
        if self.anomaly_detector and self.anomaly_detector.model:
            self.anomaly_detector.model.save(self.model_path)
            state_path = self.model_path.replace('.keras', '_state.pkl')
            self.anomaly_detector.save_state(state_path)
            print(f"Model and state saved to {self.model_path} and {state_path}")
    
    def analyze_games(self, sequences, metadata):
        results = []
        for i, ((sequence, mask), meta) in enumerate(zip(sequences, metadata)):
            detection_result = self.anomaly_detector.detect_anomalies(sequence, mask, self.feature_names)
            suspicious_moves_info = []
            for move_idx in detection_result['suspicious_moves']:
                if move_idx < len(meta['move_data']):
                    move_data = meta['move_data'][move_idx]
                    move_no = move_data['Move No.']
                    player = "White" if move_no % 2 == 1 else "Black"
                    z_score = detection_result['move_z_scores'].get(move_idx, 0)
                    error_val = detection_result['move_errors'].get(move_idx, 0)
                    his_idx = self.feature_names.index('Human_Inaccuracy_Score') if 'Human_Inaccuracy_Score' in self.feature_names else -1
                    time_ratio_idx = self.feature_names.index('Time_Ratio') if 'Time_Ratio' in self.feature_names else -1
                    his_score = sequence[move_idx, his_idx] if his_idx >= 0 else 0
                    time_ratio = sequence[move_idx, time_ratio_idx] if time_ratio_idx >= 0 else 0
                    suspicious_moves_info.append({
                        'Move No.': int(move_no),
                        'Player': player,
                        'Move': move_data['Move'],
                        'Position': move_data['FEN Before'],
                        'Z-Score': float(z_score),
                        'Error': float(error_val),
                        'HIS': float(his_score),
                        'Threshold': float(detection_result['adaptive_threshold']),
                        'Time Ratio': float(time_ratio)
                    })
            game_result = {
                'game_id': meta['game_id'],
                'white_cheat_prob': detection_result['white_cheat_prob'],
                'black_cheat_prob': detection_result['black_cheat_prob'],
                'suspicious_moves': suspicious_moves_info,
                'perfection_score': detection_result['perfection_score'],
                'avg_anomaly_score': max(detection_result['white_avg_z'], detection_result['black_avg_z'])
            }
            results.append(game_result)
        results = self.additional_pattern_analysis(results)
        return results
    
    def additional_pattern_analysis(self, results):
        for result in results:
            suspicious_moves = result['suspicious_moves']
            if not suspicious_moves:
                continue
            engine_match_rate = sum(1 for move in suspicious_moves if move.get('HIS', 0) < 0.3)
            engine_corr_score = engine_match_rate / len(suspicious_moves) if suspicious_moves else 0
            time_ratios = [move.get('Time Ratio', 0) for move in suspicious_moves]
            time_consistency = np.std(time_ratios) if time_ratios else 0
            time_anomaly = time_consistency < 0.3
            move_numbers = [move.get('Move No.', 0) for move in suspicious_moves]
            move_numbers.sort()
            diff_moves = np.diff(move_numbers)
            burst_score = np.mean(diff_moves == 2) if len(diff_moves) > 0 else 0
            white_moves = [m for m in suspicious_moves if m.get('Player') == 'White']
            black_moves = [m for m in suspicious_moves if m.get('Player') == 'Black']
            white_burst = False
            black_burst = False
            if len(white_moves) > 2:
                white_numbers = sorted([m.get('Move No.', 0) for m in white_moves])
                white_diff = np.diff(white_numbers)
                white_burst = np.any(white_diff == 2) and len(white_moves) >= 3
            if len(black_moves) > 2:
                black_numbers = sorted([m.get('Move No.', 0) for m in black_moves])
                black_diff = np.diff(black_numbers)
                black_burst = np.any(black_diff == 2) and len(black_moves) >= 3
            result['additional_patterns'] = {
                'engine_correlation': engine_corr_score,
                'time_anomaly': time_anomaly,
                'burst_score': burst_score,
                'white_burst': white_burst,
                'black_burst': black_burst,
                'overall_pattern_score': (engine_corr_score + float(time_anomaly) + burst_score) / 3
            }
            if result['white_cheat_prob'] > 0.05:
                pattern_boost = 0
                if engine_corr_score > 0.5: pattern_boost += 0.2
                if time_anomaly: pattern_boost += 0.15
                if white_burst: pattern_boost += 0.15
                result['white_cheat_prob'] = min(0.95, result['white_cheat_prob'] + pattern_boost)
            if result['black_cheat_prob'] > 0.05:
                pattern_boost = 0
                if engine_corr_score > 0.5: pattern_boost += 0.2
                if time_anomaly: pattern_boost += 0.15
                if black_burst: pattern_boost += 0.15
                result['black_cheat_prob'] = min(0.95, result['black_cheat_prob'] + pattern_boost)
        return results
    
    def generate_report(self, results, output_file="cheating_analysis_report.txt"):
        with open(output_file, "w") as f:
            f.write("Chess Cheating Detection Report\n")
            f.write("="*50 + "\n\n")
            for i, res in enumerate(results):
                f.write(f"Game {i+1} (ID: {res['game_id']}) Analysis:\n")
                f.write("-"*50 + "\n")
                f.write(f"- White Cheating Probability: {res['white_cheat_prob']:.1%}\n")
                f.write(f"- Black Cheating Probability: {res['black_cheat_prob']:.1%}\n")
                f.write(f"- Perfection Score: {res['perfection_score']:.3f} (1.0 = clean)\n")
                f.write(f"- Avg Anomaly Score: {res['avg_anomaly_score']:.3f}\n")
                if 'additional_patterns' in res:
                    patterns = res['additional_patterns']
                    f.write("\nPattern Analysis:\n")
                    f.write(f"- Engine Correlation: {patterns['engine_correlation']:.2f}\n")
                    f.write(f"- Time Usage Anomaly: {'Yes' if patterns['time_anomaly'] else 'No'}\n")
                    f.write(f"- Suspicious Move Bursts: {patterns['burst_score']:.2f}\n")
                    if patterns['white_burst']:
                        f.write("  * Detected burst pattern in WHITE moves\n")
                    if patterns['black_burst']:
                        f.write("  * Detected burst pattern in BLACK moves\n")
                white_suspicious = res['white_cheat_prob'] >= 0.1
                black_suspicious = res['black_cheat_prob'] >= 0.1
                f.write("\nASSESSMENT: ")
                if white_suspicious and res.get('additional_patterns', {}).get('engine_correlation', 0) > 0.5:
                    f.write("Strong evidence of computer assistance for WHITE player\n")
                elif black_suspicious and res.get('additional_patterns', {}).get('engine_correlation', 0) > 0.5:
                    f.write("Strong evidence of computer assistance for BLACK player\n")
                elif white_suspicious:
                    f.write("Some suspicious patterns detected for WHITE player\n")
                elif black_suspicious:
                    f.write("Some suspicious patterns detected for BLACK player\n")
                else:
                    f.write("No significant evidence of cheating detected\n")
                if res['suspicious_moves']:
                    f.write("\nSuspicious Moves Detected:\n")
                    for move in res['suspicious_moves']:
                        f.write(f"\n  Move {move['Move No.']} ({move['Player']}):\n")
                        f.write(f"   - HIS Score: {move['HIS']:.3f} (Threshold: {move['Threshold']:.3f})\n")
                        f.write(f"   - Time Ratio: {move['Time Ratio']:.2f}\n")
                        f.write(f"   - Position: {move['Position']}\n")
                        f.write(f"   - Move Played: {move['Move']}\n")
                else:
                    f.write("\nNo suspicious moves detected\n")
                f.write("="*50 + "\n\n")

# Rest of the code (unchanged)
def run_chess_cheating_detection(data_path, output_file="cheating_analysis_report3.txt", train=True, force_fit_scalers=False):
    detector = ChessCheatingDetector()
    print("Preparing data...")
    sequences, metadata = detector.prepare_data(data_path, train=train, force_fit_scalers=force_fit_scalers)
    if train:
        print("Training model...")
        detector.train_model(sequences, epochs=30, batch_size=16)
        detector.save_model()
    else:
        print("Loading pre-trained model...")
        detector.load_model()
    print("Analyzing games for potential cheating...")
    results = detector.analyze_games(sequences, metadata)
    print("Generating report...")
    detector.generate_report(results, output_file)
    print("Visualizing results...")
    visualize_analysis(results)
    print(f"Analysis complete. Report saved to {output_file}")
    return results

def visualize_analysis(results, num_games=5):
    if not results:
        return
    games_to_plot = min(num_games, len(results))
    fig, axs = plt.subplots(games_to_plot, 1, figsize=(12, 4*games_to_plot))
    if games_to_plot == 1:
        axs = [axs]
    for i in range(games_to_plot):
        res = results[i]
        ax = axs[i]
        white_prob = res['white_cheat_prob']
        black_prob = res['black_cheat_prob']
        labels = ['White Player', 'Black Player']
        probs = [white_prob, black_prob]
        colors = ['skyblue', 'salmon']
        bars = ax.bar(labels, probs, color=colors)
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Suspicious Threshold')
        ax.set_ylabel('Cheating Probability')
        ax.set_title(f'Game {i+1} (ID: {res["game_id"]}) Cheating Analysis')
        ax.set_ylim(0, max(1.0, max(probs) * 1.2))
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.2%}', ha='center', va='bottom')
        suspicious_count = len(res['suspicious_moves'])
        ax.text(0.5, 0.9, f'Suspicious Moves: {suspicious_count}', 
                transform=ax.transAxes, ha='center', 
                bbox=dict(facecolor='yellow', alpha=0.2))
    plt.tight_layout()
    plt.savefig('cheating_analysis_visualization.png')
    plt.close()

if __name__ == "__main__":
    data_path = "blitz_lstm_human_data_2275_2325.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
    else:
        results = run_chess_cheating_detection(data_path, train=True, force_fit_scalers= True)