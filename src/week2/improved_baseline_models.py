"""
Improved Baseline Clustering Models for RFM Customer Segmentation
Fixed version with better segment naming and evaluation
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
    Improved baseline clustering model for RFM customer segmentation
    Implements K-Means with better segment naming and evaluation
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
        print(f"   Calinski-Harabasz Score: {results['calinski_harabasz'][best_idx]:.1f}")
        
        # Interpret quality
        self._interpret_quality(results['silhouette'][best_idx], 
                               results['davies_bouldin'][best_idx],
                               results['calinski_harabasz'][best_idx])
        
        # Train final model with best k
        self.best_model = KMeans(n_clusters=self.best_k, random_state=42, n_init=10)
        self.labels = self.best_model.fit_predict(X)
        self.X_scaled = X
        
        return self.best_model, results
    
    def _interpret_quality(self, silhouette, davies_bouldin, calinski_harabasz):
        """
        Interpret clustering quality metrics
        """
        print(f"\n📊 Clustering Quality Assessment:")
        
        # Silhouette interpretation
        if silhouette > 0.5:
            sil_quality = "EXCELLENT ✅ - Clusters are well-separated"
        elif silhouette > 0.35:
            sil_quality = "GOOD ✅ - Clear cluster structure"
        elif silhouette > 0.25:
            sil_quality = "ACCEPTABLE ⚠️ - Moderate separation"
        else:
            sil_quality = "POOR ❌ - Weak or overlapping clusters"
        print(f"   Silhouette ({silhouette:.3f}): {sil_quality}")
        
        # Davies-Bouldin interpretation
        if davies_bouldin < 1.0:
            db_quality = "EXCELLENT ✅ - Clusters are distinct"
        elif davies_bouldin < 1.5:
            db_quality = "GOOD ✅ - Well-separated clusters"
        elif davies_bouldin < 2.0:
            db_quality = "ACCEPTABLE ⚠️ - Some overlap"
        else:
            db_quality = "POOR ❌ - Clusters too similar"
        print(f"   Davies-Bouldin ({davies_bouldin:.3f}): {db_quality}")
        
        # Calinski-Harabasz interpretation
        if calinski_harabasz > 1000:
            ch_quality = "EXCELLENT ✅ - Strong variance ratio"
        elif calinski_harabasz > 500:
            ch_quality = "GOOD ✅ - Good variance ratio"
        elif calinski_harabasz > 200:
            ch_quality = "ACCEPTABLE ⚠️ - Moderate variance ratio"
        else:
            ch_quality = "POOR ❌ - Weak variance ratio"
        print(f"   Calinski-Harabasz ({calinski_harabasz:.1f}): {ch_quality}")
        
        # Overall recommendation
        print(f"\n💡 Overall Assessment:")
        if silhouette > 0.35 and davies_bouldin < 1.5:
            print("   ✅ Segmentation is reliable and actionable")
            print("   ✅ Transfer learning potential: HIGH")
        elif silhouette > 0.25 and davies_bouldin < 2.0:
            print("   ⚠️  Segmentation is acceptable but could be improved")
            print("   ⚠️  Transfer learning potential: MODERATE (needs fine-tuning)")
        else:
            print("   ❌ Segmentation quality is poor")
            print("   ❌ Transfer learning potential: LOW (new model recommended)")
    
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
        Get RFM profiles for each customer segment with improved naming
        
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
        
        # Add improved segment names
        profiles['Segment_Name'] = profiles.index.map(
            lambda x: self._name_segment_improved(
                df_segments[df_segments['Segment'] == x], 
                df_segments
            )
        )
        
        # Add segment quality scores
        profiles['Value_Score'] = profiles.index.map(
            lambda x: self._calculate_value_score(
                df_segments[df_segments['Segment'] == x],
                df_segments
            )
        )
        
        # Sort by value score (best segments first)
        profiles = profiles.sort_values('Value_Score', ascending=False)
        
        return profiles
    
    def _calculate_value_score(self, segment_df, full_df):
        """
        Calculate a value score for each segment (0-100)
        Higher score = more valuable customers
        """
        # Normalize metrics (0-1 scale, inverted for Recency)
        r_score = 1 - (segment_df['Recency'].mean() / full_df['Recency'].max())
        f_score = segment_df['Frequency'].mean() / full_df['Frequency'].max()
        m_score = segment_df['Monetary'].mean() / full_df['Monetary'].max()
        
        # Weighted combination (Monetary weighted highest)
        value_score = (r_score * 0.25 + f_score * 0.35 + m_score * 0.40) * 100
        
        return round(value_score, 1)
    
    def _name_segment_improved(self, segment_df, full_df):
        """
        Improved segment naming with clearer business logic
        
        Parameters:
        -----------
        segment_df : pandas.DataFrame
            Dataframe for a specific segment
        full_df : pandas.DataFrame
            Full dataset for population-level comparisons
            
        Returns:
        --------
        name : str
            Descriptive segment name with emoji
        """
        # Segment averages
        avg_r = segment_df['Recency'].mean()
        avg_f = segment_df['Frequency'].mean()
        avg_m = segment_df['Monetary'].mean()
        
        # Population quartiles for better classification
        r_25 = full_df['Recency'].quantile(0.25)
        r_50 = full_df['Recency'].quantile(0.50)
        r_75 = full_df['Recency'].quantile(0.75)
        
        f_25 = full_df['Frequency'].quantile(0.25)
        f_50 = full_df['Frequency'].quantile(0.50)
        f_75 = full_df['Frequency'].quantile(0.75)
        
        m_25 = full_df['Monetary'].quantile(0.25)
        m_50 = full_df['Monetary'].quantile(0.50)
        m_75 = full_df['Monetary'].quantile(0.75)
        
        # Classify RFM levels (Low recency is GOOD)
        r_level = 'Excellent' if avg_r < r_25 else ('Good' if avg_r < r_50 else ('Fair' if avg_r < r_75 else 'Poor'))
        f_level = 'Excellent' if avg_f > f_75 else ('Good' if avg_f > f_50 else ('Fair' if avg_f > f_25 else 'Poor'))
        m_level = 'Excellent' if avg_m > m_75 else ('Good' if avg_m > m_50 else ('Fair' if avg_m > m_25 else 'Poor'))
        
        # IMPROVED NAMING LOGIC
        
        # 1. CHAMPIONS: Best customers (Recent + High F + High M)
        if r_level in ['Excellent', 'Good'] and f_level in ['Excellent', 'Good'] and m_level in ['Excellent', 'Good']:
            return "🌟 Champions"
        
        # 2. LOYAL CUSTOMERS: Frequent buyers, decent value
        elif r_level in ['Excellent', 'Good'] and f_level in ['Excellent', 'Good']:
            return "💎 Loyal Customers"
        
        # 3. WHALES: Big spenders but infrequent
        elif m_level == 'Excellent' and f_level in ['Poor', 'Fair']:
            return "🐋 Big Spenders"
        
        # 4. PROMISING: New/recent customers with potential
        elif r_level == 'Excellent' and f_level in ['Fair', 'Poor'] and m_level in ['Good', 'Fair']:
            return "🌱 Promising"
        
        # 5. AT RISK: Were good, now declining
        elif r_level in ['Fair', 'Poor'] and f_level in ['Excellent', 'Good'] and m_level in ['Excellent', 'Good']:
            return "⚠️ At Risk - High Value"
        
        # 6. NEED ATTENTION: Decent history but not recent
        elif r_level in ['Fair', 'Poor'] and f_level in ['Good', 'Fair']:
            return "🔔 Need Attention"
        
        # 7. HIBERNATING: Long time since purchase, low activity
        elif r_level == 'Poor' and f_level in ['Poor', 'Fair']:
            return "😴 Hibernating"
        
        # 8. NEW CUSTOMERS: Recent, low frequency, low spend
        elif r_level in ['Excellent', 'Good'] and f_level == 'Poor' and m_level == 'Poor':
            return "🆕 New Customers"
        
        # 9. ABOUT TO SLEEP: Medium recency, declining
        elif r_level == 'Fair' and f_level == 'Poor' and m_level == 'Poor':
            return "💤 About to Sleep"
        
        # 10. LOST: Very old, minimal activity
        elif r_level == 'Poor' and f_level == 'Poor' and m_level == 'Poor':
            return "❌ Lost Customers"
        
        # Default: Standard with RFM indicators
        else:
            return f"📊 Standard ({r_level[0]}{f_level[0]}{m_level[0]})"
    
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
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each segment
        unique_labels = sorted(set(self.labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
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
                      c=[color], label=label_name, alpha=0.7, s=60, marker=marker, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Recency (days)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency (purchases)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Monetary (₹)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved 3D plot: {save_path}")
        
        plt.show()
    
    def plot_segment_profiles(self, profiles, title="Segment RFM Profiles", save_path=None):
        """
        Visualize RFM profiles for each segment with value scores
        
        Parameters:
        -----------
        profiles : pandas.DataFrame
            Segment profiles from get_segment_profiles()
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        # Prepare data
        segment_names = [name.split(' ', 1)[1] if ' ' in name else name for name in profiles['Segment_Name']]
        colors = plt.cm.Set3(np.linspace(0, 1, len(segment_names)))
        
        # Plot 1: Recency
        axes[0, 0].barh(segment_names, profiles['Recency_mean'], color=colors, alpha=0.8, edgecolor='black')
        axes[0, 0].set_xlabel('Avg Recency (days)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Recency by Segment (Lower is Better)', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        axes[0, 0].invert_yaxis()
        
        # Plot 2: Frequency
        axes[0, 1].barh(segment_names, profiles['Frequency_mean'], color=colors, alpha=0.8, edgecolor='black')
        axes[0, 1].set_xlabel('Avg Frequency (purchases)', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Frequency by Segment (Higher is Better)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        axes[0, 1].invert_yaxis()
        
        # Plot 3: Monetary
        axes[1, 0].barh(segment_names, profiles['Monetary_mean'], color=colors, alpha=0.8, edgecolor='black')
        axes[1, 0].set_xlabel('Avg Monetary (₹)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Monetary by Segment (Higher is Better)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        axes[1, 0].invert_yaxis()
        
        # Plot 4: Value Score
        axes[1, 1].barh(segment_names, profiles['Value_Score'], color=colors, alpha=0.8, edgecolor='black')
        axes[1, 1].set_xlabel('Customer Value Score (0-100)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Overall Segment Value (Higher is Better)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        axes[1, 1].invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(profiles['Value_Score']):
            axes[1, 1].text(v + 1, i, f'{v:.0f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved profile plot: {save_path}")
        
        plt.show()
    
    def plot_segment_distribution(self, profiles, title="Customer Distribution", save_path=None):
        """
        Plot customer distribution across segments with value indication
        
        Parameters:
        -----------
        profiles : pandas.DataFrame
            Segment profiles
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Prepare data
        sizes = profiles['Recency_count']
        labels = [f"{name}\n({int(size)} customers)" 
                 for name, size in zip(profiles['Segment_Name'], sizes)]
        colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
        
        # Pie chart with value indication
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                             colors=colors, startangle=90,
                                             textprops={'fontsize': 9, 'weight': 'bold'})
        ax1.set_title('Customer Distribution by Segment', fontsize=12, fontweight='bold', pad=20)
        
        # Bar chart with value scores
        segment_names = [name.split(' ', 1)[1] if ' ' in name else name for name in profiles['Segment_Name']]
        x = np.arange(len(segment_names))
        
        bars = ax2.bar(x, sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Segment', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
        ax2.set_title('Customer Count by Segment', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(segment_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value scores as text on bars
        for i, (bar, value) in enumerate(zip(bars, profiles['Value_Score'])):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(sizes)*0.01,
                    f'Value: {value:.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
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
    Plot elbow curve for optimal k selection with enhanced visualization
    
    Parameters:
    -----------
    results : dict
        Results from find_optimal_k method
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Optimal K Selection - Multiple Metrics', fontsize=16, fontweight='bold')
    
    k_values = results['k']
    
    # 1. Elbow curve (Inertia)
    axes[0, 0].plot(k_values, results['inertia'], 'bo-', linewidth=2, markersize=10)
    axes[0, 0].set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Elbow Method - Inertia', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(k_values)
    
    # 2. Silhouette score
    best_idx = np.argmax(results['silhouette'])
    best_k = results['k'][best_idx]
    
    axes[0, 1].plot(k_values, results['silhouette'], 'ro-', linewidth=2, markersize=10)
    axes[0, 1].axvline(x=best_k, color='green', linestyle='--', linewidth=2,
                       label=f'Best k={best_k} (Score={results["silhouette"][best_idx]:.3f})')
    axes[0, 1].axhline(y=0.5, color='orange', linestyle=':', linewidth=1.5, label='Excellent (>0.5)')
    axes[0, 1].axhline(y=0.35, color='yellow', linestyle=':', linewidth=1.5, label='Good (>0.35)')
    axes[0, 1].axhline(y=0.25, color='red', linestyle=':', linewidth=1.5, label='Acceptable (>0.25)')
    axes[0, 1].set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Silhouette Score vs k (Higher is Better)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_xticks(k_values)
    
    # 3. Davies-Bouldin Index
    axes[1, 0].plot(k_values, results['davies_bouldin'], 'go-', linewidth=2, markersize=10)
    axes[1, 0].axhline(y=1.0, color='orange', linestyle=':', linewidth=1.5, label='Excellent (<1.0)')
    axes[1, 0].axhline(y=1.5, color='yellow', linestyle=':', linewidth=1.5, label='Good (<1.5)')
    axes[1, 0].axhline(y=2.0, color='red', linestyle=':', linewidth=1.5, label='Acceptable (<2.0)')
    axes[1, 0].set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Davies-Bouldin Index', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Davies-Bouldin Index vs k (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_xticks(k_values)
    
    # 4. Calinski-Harabasz Score
    axes[1, 1].plot(k_values, results['calinski_harabasz'], 'mo-', linewidth=2, markersize=10)
    axes[1, 1].axhline(y=1000, color='orange', linestyle=':', linewidth=1.5, label='Excellent (>1000)')
    axes[1, 1].axhline(y=500, color='yellow', linestyle=':', linewidth=1.5, label='Good (>500)')
    axes[1, 1].axhline(y=200, color='red', linestyle=':', linewidth=1.5, label='Acceptable (>200)')
    axes[1, 1].set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Calinski-Harabasz Score', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Calinski-Harabasz Score vs k (Higher is Better)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_xticks(k_values)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved elbow curve: {save_path}")
    
    plt.show()