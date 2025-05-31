import pandas as pd
from MSLSTM import ChessCheatingDetector

def usage_method(elo_range, format, input_csv_path, output_csv_path):
    """
    Load a pre-trained ChessCheatingDetector model based on Elo range and format, process input CSV,
    and save analysis results to an output CSV.
    
    Parameters:
    - elo_range (str): Elo range of the model (e.g., '2475_2525')
    - format (str): Game format (e.g., 'Bullet')
    - input_csv_path (str): Path to input CSV with game data
    - output_csv_path (str): Path to save output CSV with analysis results
    """
    # Construct model and scaler file paths based on Elo range and format
    model_path = f"Models/MSLSTM_detector_{format.lower()}_{elo_range}.keras"
    scaler_file = f"Models/Scaling/lstm_{format.lower()}_{elo_range}_scalers.pkl"
    
    # Initialize the ChessCheatingDetector with the specified model and scaler files
    detector = ChessCheatingDetector(max_seq_len=200, scaler_file=scaler_file, model_path=model_path)
    
    # Prepare the data for analysis (train=False to load scalers)
    sequences, metadata = detector.prepare_data(input_csv_path, train=False)
    
    # Load the pre-trained model
    detector.load_model()
    
    # Analyze the games for potential cheating
    results = detector.analyze_games(sequences, metadata)
    
    # Create a DataFrame from the analysis results
    results_df = pd.DataFrame([{
        'Game ID': res['game_id'],
        'White Cheat Prob': res['white_cheat_prob'],
        'Black Cheat Prob': res['black_cheat_prob'],
        'Perfection Score': res['perfection_score'],
        'Avg Anomaly Score': res['avg_anomaly_score'],
        'Engine Correlation': res.get('additional_patterns', {}).get('engine_correlation', 0),
        'Time Anomaly': res.get('additional_patterns', {}).get('time_anomaly', False),
        'Burst Score': res.get('additional_patterns', {}).get('burst_score', 0),
        'White Burst': res.get('additional_patterns', {}).get('white_burst', False),
        'Black Burst': res.get('additional_patterns', {}).get('black_burst', False),
        'Overall Pattern Score': res.get('additional_patterns', {}).get('overall_pattern_score', 0)
    } for res in results])
    
    # Save the results to the specified output CSV file
    results_df.to_csv(output_csv_path, index=False)
    
    print(f"Analysis complete. Results saved to {output_csv_path}")

# Example usage
if __name__ == "__main__":
#    usage_method('2275_2325', 'Blitz', 'Data/blitz/SC_lstm_human_data_2275_2325.csv', 'Evaluted/Eval_SC_MSLSTM_Blitz_2300.csv')
    usage_method('2275_2325', 'Bullet', 'Data/HC_lstm_human_data_2275_2325.csv', 'Evaluted/Eval_HC_MSLSTM_Bullet_2300.csv')
#    usage_method('2275_2325', 'Blitz', 'Data/blitz/clean_lstm_human_data_2275_2325.csv', 'Evaluted/Eval_Clean_MSLSTM_Blitz_2300.csv')