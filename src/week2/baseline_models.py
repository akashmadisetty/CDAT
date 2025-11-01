"""
Baseline Clustering Models for RFM Customer Segmentation
Member 2 - Week 2 Deliverable
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pickle
import warnings
warnings.filterwarnings('ignore')


class RFMSegmentationModel:
    """
    Baseline clustering model for RFM customer segmentation
    Implements K-Means, DBSCAN, and Hierarchical clustering
    """
    
    def __init__(self, rfm_features=['Recency', 'Frequency', 'Monetary']):
        """
        Initialize the RFM segmentation model
        
        Parameters:
        -----------
        rfm_features : list
            List of RFM feature column names to use for clustering
        """
        self.rfm_features = rfm_features
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_k = None
        self.best_score = -1
        self.X_scaled = None
        self.labels = None
        self.customer_ids = None
        
    def prepare_data(self, df):
        """
        Prepare and scale RFM data for clustering
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with RFM data
            
        Returns:
        --------
        X_scaled : numpy.ndarray
            Scaled RFM feature matrix
        customer_ids : array
            Customer IDs corresponding to rows
        """
        # Handle missing values
        df_clean = df[['customer_id'] + self.rfm_features].copy()
        df_clean = df_clean.dropna()
        
        # Store customer IDs
        self.customer_ids = df_clean['customer_id'].values
        
        # Extract RFM features
        X = df_clean[self.rfm_features].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✓ Data prepared: {X_scaled.shape[0]} customers, {X_scaled.shape[1]} RFM features")
        return X_scaled, self.customer_ids
    
    def find_optimal_k(self, X, k_range=[3, 4, 5, 6, 7, 8]):
        """
        Find optimal number of clusters using Elbow method and Silhouette score
        
        Parameters:
        -----------
        X : numpy.ndarray
            Scaled RFM feature matrix
        k_range : list
            Range of k values to test
            
        Returns:
        --------
        results : dict
            Dictionary containing metrics for each k
        """
        results = {
            'k': [],
            'inertia': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        print(f"\n🔍 Finding optimal k in range {k_range}...")
        
        for k in k_range:
            # Train K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Calculate metrics
            inertia = kmeans.inertia_
            silhouette = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)
            
            results['k'].append(k)
            results['inertia'].append(inertia)
            results['silhouette'].append(silhouette)
            results['davies_bouldin'].append(db_score)
            results['calinski_harabasz'].append(ch_score)
            
            print(f"  k={k}: Silhouette={silhouette:.3f}, Davies-Bouldin={db_score:.3f}, CH={ch_score:.1f}")
        
        return results
    
    def train_kmeans(self, X, k_range=[3, 4, 5, 6, 7, 8]):
        """
        Train K-Means model and find optimal k
        
        Parameters:
        -----------
        X : numpy.ndarray
            Scaled RFM feature matrix
        k_range : list
            Range of k values to test
            
        Returns:
        --------
        best_model : KMeans
            Trained model with optimal k
        results : dict
            Optimization results
        """
        # Find optimal k
        results = self.find_optimal_k(X, k_range)
        
        # Select best k based on silhouette score
        best_idx = np.argmax(results['silhouette'])
        self.best_k = results['k'][best_idx]
        
        print(f"\n✅ Optimal k selected: {self.best_k}")
        print(f"   Silhouette Score: {results['silhouette'][best_idx]:.3f}")
        print(f"   Davies-Bouldin Index: {results['davies_bouldin'][best_idx]:.3f}")
        
        # Train final model with best k
        self.best_model = KMeans(n_clusters=self.best_k, random_state=42, n_init=10)
        self.labels = self.best_model.fit_predict(X)
        self.X_scaled = X
        
        return self.best_model, results
    
    def evaluate(self, model, X, labels=None):
        """
        Evaluate clustering model with multiple metrics
        
        Parameters:
        -----------
        model : sklearn clustering model
            Trained clustering model
        X : numpy.ndarray
            Scaled RFM feature matrix
        labels : numpy.ndarray, optional
            Cluster labels (if already computed)
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        if labels is None:
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
        
        # Handle DBSCAN noise points
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]
        
        metrics = {}
        
        # Silhouette Score (higher is better, range: -1 to 1)
        if len(set(labels_filtered)) > 1:
            metrics['silhouette_score'] = silhouette_score(X_filtered, labels_filtered)
        else:
            metrics['silhouette_score'] = -1
        
        # Davies-Bouldin Index (lower is better)
        if len(set(labels_filtered)) > 1:
            metrics['davies_bouldin_index'] = davies_bouldin_score(X_filtered, labels_filtered)
        else:
            metrics['davies_bouldin_index'] = np.inf
        
        # Calinski-Harabasz Score (higher is better)
        if len(set(labels_filtered)) > 1:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_filtered, labels_filtered)
        else:
            metrics['calinski_harabasz_score'] = 0
        
        # Cluster size distribution
        unique, counts = np.unique(labels_filtered, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique, counts))
        metrics['n_clusters'] = len(unique)
        metrics['n_noise'] = np.sum(labels == -1)
        
        # Inertia (for K-Means only)
        if hasattr(model, 'inertia_'):
            metrics['inertia'] = model.inertia_
        
        return metrics
    
    def get_segment_profiles(self, df_rfm):
        """
        Get RFM profiles for each customer segment
        
        Parameters:
        -----------
        df_rfm : pandas.DataFrame
            Original RFM dataframe
            
        Returns:
        --------
        profiles : pandas.DataFrame
            Segment profiles with RFM statistics
        """
        # Create dataframe with cluster labels
        df_segments = df_rfm.copy()
        df_segments['Segment'] = self.labels
        
        # Calculate profiles
        profiles = df_segments.groupby('Segment')[self.rfm_features].agg(['mean', 'median', 'std', 'count'])
        profiles.columns = ['_'.join(col).strip() for col in profiles.columns.values]
        
        # Add segment names based on RFM characteristics
        # Pass the FULL dataset for proper comparison
        profiles['Segment_Name'] = profiles.index.map(
            lambda x: self._name_segment(
                df_segments[df_segments['Segment'] == x], 
                df_segments  # Pass full dataset for quantile comparison
            )
        )
        
        return profiles
    
    def _name_segment(self, segment_df, full_df):
        """
        Assign descriptive name to segment based on RFM characteristics
        Compares segment averages to OVERALL population quantiles
        
        Parameters:
        -----------
        segment_df : pandas.DataFrame
            Dataframe for a specific segment
        full_df : pandas.DataFrame
            Full dataset for population-level comparisons
            
        Returns:
        --------
        name : str
            Descriptive segment name
        """
        # Segment averages
        avg_r = segment_df['Recency'].mean()
        avg_f = segment_df['Frequency'].mean()
        avg_m = segment_df['Monetary'].mean()
        
        # Population quantiles (33rd and 66th percentiles)
        r_33 = full_df['Recency'].quantile(0.33)
        r_66 = full_df['Recency'].quantile(0.66)
        f_33 = full_df['Frequency'].quantile(0.33)
        f_66 = full_df['Frequency'].quantile(0.66)
        m_33 = full_df['Monetary'].quantile(0.33)
        m_66 = full_df['Monetary'].quantile(0.66)
        
        # Classify segment based on RFM levels
        r_level = 'Low' if avg_r < r_33 else ('High' if avg_r > r_66 else 'Med')
        f_level = 'High' if avg_f > f_66 else ('Low' if avg_f < f_33 else 'Med')
        m_level = 'High' if avg_m > m_66 else ('Low' if avg_m < m_33 else 'Med')
        
        # Champions: Recent, Frequent, High-value
        if r_level == 'Low' and f_level == 'High' and m_level == 'High':
            return "Champions"
        
        # Loyal Customers: Recent, Frequent, but not necessarily high-value
        elif r_level == 'Low' and f_level == 'High':
            return "Loyal Customers"
        
        # At Risk: Not recent but used to be good
        elif r_level == 'High' and f_level == 'High' and m_level == 'High':
            return "At Risk High Value"
        
        # Promising: Recent, decent spend, but low frequency (new customers with potential)
        elif r_level == 'Low' and m_level in ['Med', 'High'] and f_level == 'Low':
            return "Promising"
        
        # Hibernating: Haven't purchased in a while
        elif r_level == 'High' and f_level == 'Low':
            return "Hibernating"
        
        # Need Attention: Frequent buyers who haven't returned recently
        elif r_level == 'High' and f_level in ['Med', 'High']:
            return "Need Attention"
        
        # New Customers: Recent but low frequency and low spend
        elif r_level == 'Low' and f_level == 'Low' and m_level == 'Low':
            return "New Customers"
        
        # About to Sleep: Medium recency, declining
        elif r_level == 'Med' and f_level == 'Low' and m_level == 'Low':
            return "About to Sleep"
        
        # Standard/Average: Everything else
        else:
            return f"Standard ({r_level[0]}{f_level[0]}{m_level[0]})"
    
    def plot_rfm_3d(self, df_rfm, title="RFM 3D Scatter", save_path=None):
        """
        Visualize segments in 3D RFM space
        
        Parameters:
        -----------
        df_rfm : pandas.DataFrame
            Original RFM dataframe
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each segment
        unique_labels = set(self.labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'gray'
                marker = 'x'
                label_name = 'Noise'
            else:
                marker = 'o'
                label_name = f'Segment {label}'
            
            mask = self.labels == label
            ax.scatter(df_rfm.loc[mask, 'Recency'],
                      df_rfm.loc[mask, 'Frequency'],
                      df_rfm.loc[mask, 'Monetary'],
                      c=[color], label=label_name, alpha=0.6, s=50, marker=marker)
        
        ax.set_xlabel('Recency (days)', fontsize=10)
        ax.set_ylabel('Frequency (purchases)', fontsize=10)
        ax.set_zlabel('Monetary (₹)', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved 3D plot: {save_path}")
        
        plt.show()
    
    def plot_segment_profiles(self, profiles, title="Segment RFM Profiles", save_path=None):
        """
        Visualize RFM profiles for each segment
        
        Parameters:
        -----------
        profiles : pandas.DataFrame
            Segment profiles from get_segment_profiles()
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Extract mean values for each RFM metric
        segments = profiles.index
        recency = profiles['Recency_mean']
        frequency = profiles['Frequency_mean']
        monetary = profiles['Monetary_mean']
        
        # Plot Recency
        axes[0].bar(segments, recency, color='coral', alpha=0.7)
        axes[0].set_xlabel('Segment', fontsize=11)
        axes[0].set_ylabel('Avg Recency (days)', fontsize=11)
        axes[0].set_title('Recency by Segment', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Plot Frequency
        axes[1].bar(segments, frequency, color='skyblue', alpha=0.7)
        axes[1].set_xlabel('Segment', fontsize=11)
        axes[1].set_ylabel('Avg Frequency (purchases)', fontsize=11)
        axes[1].set_title('Frequency by Segment', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Plot Monetary
        axes[2].bar(segments, monetary, color='lightgreen', alpha=0.7)
        axes[2].set_xlabel('Segment', fontsize=11)
        axes[2].set_ylabel('Avg Monetary (₹)', fontsize=11)
        axes[2].set_title('Monetary by Segment', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved profile plot: {save_path}")
        
        plt.show()
    
    def plot_segment_distribution(self, profiles, title="Customer Distribution", save_path=None):
        """
        Plot customer distribution across segments
        
        Parameters:
        -----------
        profiles : pandas.DataFrame
            Segment profiles
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart
        sizes = profiles['Recency_count']
        labels = [f"Segment {i}\n({profiles.loc[i, 'Segment_Name']})" 
                 for i in profiles.index]
        colors = plt.cm.Spectral(np.linspace(0, 1, len(sizes)))
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Customer Distribution by Segment', fontsize=12)
        
        # Bar chart
        ax2.bar(profiles.index, sizes, color=colors, alpha=0.7)
        ax2.set_xlabel('Segment', fontsize=11)
        ax2.set_ylabel('Number of Customers', fontsize=11)
        ax2.set_title('Customer Count by Segment', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved distribution plot: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'rfm_features': self.rfm_features,
            'best_k': self.best_k,
            'labels': self.labels,
            'customer_ids': self.customer_ids
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved: {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.rfm_features = model_data['rfm_features']
        self.best_k = model_data['best_k']
        self.labels = model_data['labels']
        self.customer_ids = model_data.get('customer_ids')
        
        print(f"✓ Model loaded: {filepath}")


def plot_elbow_curve(results, save_path=None):
    """
    Plot elbow curve for optimal k selection
    
    Parameters:
    -----------
    results : dict
        Results from find_optimal_k method
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow curve (Inertia)
    axes[0].plot(results['k'], results['inertia'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[0].set_ylabel('Inertia', fontsize=11)
    axes[0].set_title('Elbow Method', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette score
    axes[1].plot(results['k'], results['silhouette'], 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].set_title('Silhouette Score vs k', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Mark best k
    best_idx = np.argmax(results['silhouette'])
    best_k = results['k'][best_idx]
    axes[1].axvline(x=best_k, color='green', linestyle='--', linewidth=2,
                    label=f'Best k={best_k}')
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved elbow curve: {save_path}")
    
    plt.show()