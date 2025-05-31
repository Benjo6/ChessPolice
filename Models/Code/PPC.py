import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ChessPlayerClustering:
    """
    Clustering model for chess player analysis using statistical clustering methods
    """
    
    def __init__(self, max_clusters=10, random_state=42):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.kmeans = None
        self.isolation_forest = None
        self.cluster_centers = None
        self.feature_names = None
        self.human_indices = {}
        self.distance_thresholds = {}
        self.best_n_clusters = 4
        self.random_state = random_state

    def preprocess_data(self, df):
        """Preprocess and normalize the data"""
        non_features = ['Player ID', 'Game Format']
        features = [col for col in df.columns if col not in non_features]
        self.feature_names = features
        data = self.imputer.fit_transform(df[features])
        return self.scaler.fit_transform(data)

    def find_optimal_clusters(self, data, max_clusters=10):
        """Determine optimal number of clusters using multiple metrics"""
        scores = {'silhouette': [], 'calinski': []}
        for n in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(data)
            if len(np.unique(labels)) < 2:
                scores['silhouette'].append(0)
                scores['calinski'].append(0)
                continue
            scores['silhouette'].append(silhouette_score(data, labels))
            scores['calinski'].append(calinski_harabasz_score(data, labels))
        sil_norm = np.array(scores['silhouette']) / np.max(scores['silhouette'])
        cal_norm = np.array(scores['calinski']) / np.max(scores['calinski'])
        combined = 0.6 * sil_norm + 0.4 * cal_norm
        self.best_n_clusters = np.argmax(combined) + 2
        return self.best_n_clusters

    def fit(self, df):
        """Main training method"""
        processed_data = self.preprocess_data(df)
        self.find_optimal_clusters(processed_data)
        print(f"Optimal clusters found: {self.best_n_clusters}")
        self.kmeans = KMeans(n_clusters=self.best_n_clusters, 
                            random_state=self.random_state, n_init=10)
        clusters = self.kmeans.fit_predict(processed_data)
        self.cluster_centers = self.kmeans.cluster_centers_
        self._calculate_cluster_stats(processed_data, clusters, df)
        self.isolation_forest = IsolationForest(contamination=0.05, 
                                               random_state=self.random_state)
        self.isolation_forest.fit(processed_data)
        self._calculate_distance_thresholds(processed_data, clusters)
        return self

    def _calculate_cluster_stats(self, data, clusters, original_df):
        """Calculate cluster statistics and human indices"""
        cluster_df = pd.DataFrame(data, columns=self.feature_names)
        cluster_df['Cluster'] = clusters
        performance_metrics = ['Average Player Elo', 'Avg MM', 'Avg EV',
                              'Win/Loss Ratio', 'Critical Move Accuracy']
        metrics = [m for m in performance_metrics if m in self.feature_names]
        agg_scores = cluster_df.groupby('Cluster')[metrics].mean().mean(axis=1)
        sorted_clusters = agg_scores.sort_values().index.tolist()
        self.human_indices = {cluster: idx + 1 for idx, cluster in enumerate(sorted_clusters)}

    def _calculate_distance_thresholds(self, data, clusters):
        """Calculate distance thresholds for each cluster"""
        for cluster_id in range(self.best_n_clusters):
            cluster_data = data[clusters == cluster_id]
            centroid = self.cluster_centers[cluster_id]
            distances = np.linalg.norm(cluster_data - centroid, axis=1)
            self.distance_thresholds[cluster_id] = np.percentile(distances, 95)

    def predict_cluster(self, player_data):
        """
        Predict the cluster for a player's data using the fitted model.
        
        Parameters:
        - player_data: Input data (pd.DataFrame or NumPy array)
        
        Returns:
        - dict: Cluster assignment and related metrics
        """
        if isinstance(player_data, pd.DataFrame):
            # Use feature names from training to select columns
            features = self.feature_names
            data = self.imputer.transform(player_data[features])
            data = self.scaler.transform(data)
        else:
            # Handle NumPy array input (though not used in this solution)
            data = self.scaler.transform(self.imputer.transform(player_data))
        cluster = self.kmeans.predict(data)[0]
        centroid = self.kmeans.cluster_centers_[cluster]
        euclidean_dist = np.linalg.norm(data - centroid)
        return {
            'cluster': int(cluster),
            'human_index': self.human_indices[cluster],
            'euclidean_distance': float(euclidean_dist),
            'distance_threshold': self.distance_thresholds[cluster]
        }

    def detect_outliers(self, player_data, method='combined'):
        """
        Detect outliers in player data using the specified method.
        
        Parameters:
        - player_data: Input data (pd.DataFrame or NumPy array)
        - method: Outlier detection method ('combined', 'isolation_forest', or distance-based)
        
        Returns:
        - dict: Outlier status and related metrics
        """
        if isinstance(player_data, pd.DataFrame):
            # Use feature names from training to select columns
            features = self.feature_names
            processed_data = self.scaler.transform(self.imputer.transform(player_data[features]))
        else:
            # Handle NumPy array input (though not used in this solution)
            processed_data = self.scaler.transform(self.imputer.transform(player_data))
        prediction = self.predict_cluster(player_data)
        cluster = prediction['cluster']
        iso_pred = self.isolation_forest.predict(processed_data)
        iso_outlier = iso_pred[0] == -1
        distance_outlier = prediction['euclidean_distance'] > prediction['distance_threshold']
        if method == 'combined':
            is_outlier = iso_outlier or distance_outlier
        elif method == 'isolation_forest':
            is_outlier = iso_outlier
        else:
            is_outlier = distance_outlier
        return {
            'is_outlier': bool(is_outlier),
            'cluster': cluster,
            'human_index': prediction['human_index'],
            'isolation_forest_outlier': bool(iso_outlier),
            'distance_outlier': bool(distance_outlier),
            'distance_ratio': prediction['euclidean_distance'] / prediction['distance_threshold']
        }

    def visualize_clusters(self, df, method='PCA', save_path=None):
        """Visualize clusters using dimensionality reduction"""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        processed_data = self.preprocess_data(df)
        if method == 'PCA':
            reducer = PCA(n_components=2)
        elif method == 'TSNE':
            reducer = TSNE(n_components=2)
        else:
            raise ValueError("Use PCA or TSNE")
        reduced = reducer.fit_transform(processed_data)
        clusters = self.kmeans.predict(processed_data)
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Cluster Visualization ({method})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, path='PPC'):
        """Save model components"""
        import os
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        joblib.dump(self.imputer, f'{path}/imputer.pkl')
        joblib.dump(self.kmeans, f'{path}/kmeans.pkl')
        joblib.dump(self.isolation_forest, f'{path}/isolation_forest.pkl')
        joblib.dump({
            'human_indices': self.human_indices,
            'distance_thresholds': self.distance_thresholds,
            'feature_names': self.feature_names
        }, f'{path}/metadata.pkl')

    @classmethod
    def load_model(cls, path='PPC'):
        """Load saved model"""
        model = cls()
        model.scaler = joblib.load(f'{path}/scaler.pkl')
        model.imputer = joblib.load(f'{path}/imputer.pkl')
        model.kmeans = joblib.load(f'{path}/kmeans.pkl')
        model.isolation_forest = joblib.load(f'{path}/isolation_forest.pkl')
        metadata = joblib.load(f'{path}/metadata.pkl')
        model.human_indices = metadata['human_indices']
        model.distance_thresholds = metadata['distance_thresholds']
        model.feature_names = metadata['feature_names']
        return model


def visualize_with_new_player(model, df_train, new_player, method='PCA', save_path=None):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    features = model.feature_names
    train_data = model.imputer.transform(df_train[features])
    train_data = model.scaler.transform(train_data)
    new_data = model.imputer.transform(new_player[features])
    new_data = model.scaler.transform(new_data)
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 'TSNE':
        reducer = TSNE(n_components=2)
    else:
        raise ValueError("Method must be 'PCA' or 'TSNE'")
    reducer.fit(train_data)
    reduced_train = reducer.transform(train_data)
    reduced_new = reducer.transform(new_data)
    clusters_train = model.kmeans.predict(train_data)
    cluster_new = model.kmeans.predict(new_data)[0]
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_train[:, 0], reduced_train[:, 1],
                         c=clusters_train, cmap='tab10', alpha=0.6, label='Training Data')
    plt.scatter(reduced_new[:, 0], reduced_new[:, 1],
                c=[cluster_new], cmap='tab10', marker='*', s=200, edgecolor='k',
                label=f'New Player (Cluster {cluster_new})')
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Cluster Visualization with New Player ({method})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    df_train = pd.read_csv('blitz_clustering_player_data_2275_2325.csv')
    model = ChessPlayerClustering()
    model.fit(df_train)
    model.save_model("Models/PPC_detector_blitz_2275_2325.keras")
    model.visualize_clusters(df_train, method='PCA', save_path='cluster_visualization.png')
    new_player = pd.DataFrame([{
        'Player ID': 'jucAx',
        'Game Format': 'Bullet',
        'Number of Games': 54,
        'Average Player Elo': 2501.8051175245464,
        'Avg Time/Move': 1.118714668253496,
        'Time Pressure Frequency': 0.19696518893186551,
        'Avg MM': 0.3754834870574234,
        'Avg EV': 0.3936328473668551,
        'Avg AD': -0.23989586432609342,
        'Avg Accumulated Gain': -14.931296296296296,
        'Win/Loss Ratio': 1.0434782608695652,
        'Opening-Endgame Time Difference': -0.09739917616629945,
        'Middlegame-Endgame Time Difference': 0.8569589141017712,
        'Volatility Score': 0.8473116561950288,
        'Critical Move Accuracy': 0.3580060422960725,
        'Board Complexity Index': 0.2406608152335614
    }])
    print("Cluster Prediction:", model.predict_cluster(new_player))
    print("Outlier Detection:", model.detect_outliers(new_player))