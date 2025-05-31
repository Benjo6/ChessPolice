import pandas as pd
from sklearn.impute import SimpleImputer
from PPC import ChessPlayerClustering

def usage_method(elo_range, format, input_csv_path, output_csv_path):
    """
    Load a ChessPlayerClustering model based on Elo range and format, process input CSV,
    and save predictions to an output CSV.
    
    Parameters:
    - elo_range (str): Elo range of the model (e.g., '2475_2525')
    - format (str): Game format (e.g., 'Bullet')
    - input_csv_path (str): Path to input CSV with player data
    - output_csv_path (str): Path to save output CSV with predictions
    """
    # Construct model path based on naming convention from example
    model_path = f"Models/PPC_detector_{format.lower()}_{elo_range}.keras"
    
    # Load the pre-trained model
    model = ChessPlayerClustering.load_model(model_path)
    
    # Load input CSV
    df = pd.read_csv(input_csv_path)
    
    # Process each player and collect predictions
    # (No imputation here; the model's imputer will handle missing values)
    results = []
    for i in range(len(df)):
        # Pass a single-row DataFrame to preserve feature names
        player_data = df.iloc[[i]]  # Double brackets return a DataFrame
        prediction = model.detect_outliers(player_data)
        result = {
            'Player ID': df.iloc[i]['Player ID'],  # Identifier as per code
            'Cluster': prediction['cluster'],
            'Human Index': prediction['human_index'],
            'Is Outlier': prediction['is_outlier'],
            'Isolation Forest Outlier': prediction['isolation_forest_outlier'],
            'Distance Outlier': prediction['distance_outlier'],
            'Distance Ratio': prediction['distance_ratio']
        }
        results.append(result)
    
    # Create DataFrame from results and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)

# Example usage
usage_method('2275_2325', "Blitz", "Data/blitz/HC_clustering_player_data_2275_2325.csv", "Evaluted/Eval_HC_Clustering_Blitz_2300.csv")