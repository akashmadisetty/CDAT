"""
Transfer Learning Experiments for RFM Customer Segmentation
Member 1 - Week 3 Deliverable

Tests 5 transfer strategies:
1. Transfer as-is (0% target data)
2. Fine-tune with 10% target
3. Fine-tune with 20% target
4. Fine-tune with 50% target
5. Train from scratch (100% target)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


class TransferLearningExperiment:
    """
    Execute transfer learning experiments for customer segmentation
    """
    
    def __init__(self, pair_id, source_rfm_file, target_rfm_file, source_model_file):
        """
        Initialize experiment
        
        Parameters:
        -----------
        pair_id : int
            Domain pair identifier (1, 2, 3, or 4)
        source_rfm_file : str
            Path to source RFM data
        target_rfm_file : str
            Path to target RFM data
        source_model_file : str
            Path to trained source model (.pkl)
        """
        self.pair_id = pair_id
        self.rfm_features = ['Recency', 'Frequency', 'Monetary']
        
        # Load data
        print(f"\n📂 Loading data for Pair {pair_id}...")
        self.source_rfm = pd.read_csv(source_rfm_file)
        self.target_rfm = pd.read_csv(target_rfm_file)
        print(f"  ✓ Source: {len(self.source_rfm)} customers")
        print(f"  ✓ Target: {len(self.target_rfm)} customers")
        
        # Load source model
        print(f"  📦 Loading source model...")
        with open(source_model_file, 'rb') as f:
            model_data = pickle.load(f)
        self.source_model = model_data['model']
        self.source_scaler = model_data['scaler']
        self.source_k = model_data['best_k']
        print(f"  ✓ Source model: k={self.source_k}")
        
        # Results storage
        self.results = []
    
    def prepare_data(self, df):
        """Prepare RFM features"""
        X = df[self.rfm_features].dropna().values
        return X
    
    def evaluate_clustering(self, X, labels, test_name):
        """
        Evaluate clustering quality
        
        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        # Filter out any noise labels (-1) if present
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]
        
        metrics = {
            'test': test_name,
            'n_samples': len(X_filtered),
            'n_clusters': len(np.unique(labels_filtered))
        }
        
        # Silhouette Score
        if len(np.unique(labels_filtered)) > 1:
            metrics['silhouette'] = silhouette_score(X_filtered, labels_filtered)
        else:
            metrics['silhouette'] = -1
        
        # Davies-Bouldin Index
        if len(np.unique(labels_filtered)) > 1:
            metrics['davies_bouldin'] = davies_bouldin_score(X_filtered, labels_filtered)
        else:
            metrics['davies_bouldin'] = np.inf
        
        # Calinski-Harabasz Score
        if len(np.unique(labels_filtered)) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_filtered, labels_filtered)
        else:
            metrics['calinski_harabasz'] = 0
        
        # Cluster size distribution
        unique, counts = np.unique(labels_filtered, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
        metrics['size_std'] = np.std(counts)  # Balance metric
        metrics['size_min'] = np.min(counts)
        metrics['size_max'] = np.max(counts)
        
        return metrics
    
    def test_1_transfer_as_is(self):
        """
        Test 1: Transfer source model directly to target (no adaptation)
        """
        print("\n" + "="*80)
        print("TEST 1: TRANSFER AS-IS (0% Target Data)")
        print("="*80)
        
        # Prepare target data
        X_target = self.prepare_data(self.target_rfm)
        
        # Scale using SOURCE scaler (critical!)
        X_target_scaled = self.source_scaler.transform(X_target)
        
        # Predict using source model
        target_labels = self.source_model.predict(X_target_scaled)
        
        # Evaluate
        metrics = self.evaluate_clustering(X_target_scaled, target_labels, "Test 1: Transfer as-is")
        metrics['source_data_pct'] = 100
        metrics['target_data_pct'] = 0
        metrics['strategy'] = 'Transfer as-is'
        
        print(f"\n✅ Results:")
        print(f"   Silhouette: {metrics['silhouette']:.3f}")
        print(f"   Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
        print(f"   Clusters: {metrics['n_clusters']}")
        
        self.results.append(metrics)
        return metrics
    
    def test_2_fine_tune(self, target_pct=10):
        """
        Test 2-4: Fine-tune with X% of target data
        
        Parameters:
        -----------
        target_pct : int
            Percentage of target data to use (10, 20, or 50)
        """
        print("\n" + "="*80)
        print(f"TEST: FINE-TUNE WITH {target_pct}% TARGET DATA")
        print("="*80)
        
        # Sample target data
        n_target_samples = int(len(self.target_rfm) * (target_pct / 100))
        target_sample = self.target_rfm.sample(n=n_target_samples, random_state=42)
        
        # Combine source + target sample
        combined_data = pd.concat([self.source_rfm, target_sample], ignore_index=True)
        X_combined = self.prepare_data(combined_data)
        
        # Fit NEW scaler on combined data
        scaler = StandardScaler()
        X_combined_scaled = scaler.fit_transform(X_combined)
        
        # Re-train K-Means with source k
        print(f"  🔄 Training K-Means (k={self.source_k}) on {len(combined_data)} customers...")
        model_finetuned = KMeans(n_clusters=self.source_k, random_state=42, n_init=10)
        model_finetuned.fit(X_combined_scaled)
        
        # Evaluate on FULL target data
        X_target_full = self.prepare_data(self.target_rfm)
        X_target_full_scaled = scaler.transform(X_target_full)
        target_labels = model_finetuned.predict(X_target_full_scaled)
        
        # Evaluate
        metrics = self.evaluate_clustering(X_target_full_scaled, target_labels, 
                                          f"Test: Fine-tune {target_pct}%")
        metrics['source_data_pct'] = 100  # Always use 100% source
        metrics['target_data_pct'] = target_pct
        metrics['strategy'] = f'Fine-tune {target_pct}%'
        
        print(f"\n✅ Results:")
        print(f"   Silhouette: {metrics['silhouette']:.3f}")
        print(f"   Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
        print(f"   Clusters: {metrics['n_clusters']}")
        
        self.results.append(metrics)
        return metrics
    
    def test_5_train_from_scratch(self):
        """
        Test 5: Train new model on target data only (no transfer)
        """
        print("\n" + "="*80)
        print("TEST 5: TRAIN FROM SCRATCH (100% Target Data, No Transfer)")
        print("="*80)
        
        # Prepare target data
        X_target = self.prepare_data(self.target_rfm)
        
        # Fit new scaler
        scaler = StandardScaler()
        X_target_scaled = scaler.fit_transform(X_target)
        
        # Train NEW K-Means from scratch (use source k for fair comparison)
        print(f"  🔄 Training K-Means (k={self.source_k}) on {len(self.target_rfm)} target customers...")
        model_scratch = KMeans(n_clusters=self.source_k, random_state=42, n_init=10)
        model_scratch.fit(X_target_scaled)
        
        target_labels = model_scratch.labels_
        
        # Evaluate
        metrics = self.evaluate_clustering(X_target_scaled, target_labels, "Test 5: Train from scratch")
        metrics['source_data_pct'] = 0
        metrics['target_data_pct'] = 100
        metrics['strategy'] = 'Train from scratch'
        
        print(f"\n✅ Results:")
        print(f"   Silhouette: {metrics['silhouette']:.3f}")
        print(f"   Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
        print(f"   Clusters: {metrics['n_clusters']}")
        
        self.results.append(metrics)
        return metrics
    
    def run_all_tests(self):
        """
        Execute all 5 tests in sequence
        """
        print("\n" + "🚀"*40)
        print(f"RUNNING ALL TRANSFER LEARNING TESTS - PAIR {self.pair_id}")
        print("🚀"*40)
        
        # Test 1: Transfer as-is
        self.test_1_transfer_as_is()
        
        # Test 2: Fine-tune 10%
        self.test_2_fine_tune(target_pct=10)
        
        # Test 3: Fine-tune 20%
        self.test_2_fine_tune(target_pct=20)
        
        # Test 4: Fine-tune 50%
        self.test_2_fine_tune(target_pct=50)
        
        # Test 5: Train from scratch
        self.test_5_train_from_scratch()
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Add pair info
        results_df['pair_id'] = self.pair_id
        
        # Calculate transfer quality (vs train from scratch)
        scratch_silhouette = results_df[results_df['strategy'] == 'Train from scratch']['silhouette'].values[0]
        results_df['transfer_quality_pct'] = (results_df['silhouette'] / scratch_silhouette) * 100
        
        print("\n" + "="*80)
        print("📊 SUMMARY OF ALL TESTS")
        print("="*80)
        print(results_df[['strategy', 'silhouette', 'davies_bouldin', 'transfer_quality_pct']].to_string(index=False))
        
        print(f"\n💡 Key Findings:")
        print(f"   Best Strategy: {results_df.loc[results_df['silhouette'].idxmax(), 'strategy']}")
        print(f"   Best Silhouette: {results_df['silhouette'].max():.3f}")
        print(f"   Transfer as-is quality: {results_df[results_df['strategy']=='Transfer as-is']['transfer_quality_pct'].values[0]:.1f}%")
        
        return results_df
    
    def save_results(self, output_file):
        """
        Save results to CSV
        NOTE: This should be called AFTER adding metadata in run_experiment()
        """
        results_df = pd.DataFrame(self.results)
        results_df['pair_id'] = self.pair_id
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Saved results: {output_file}")
        return results_df


def calculate_source_baseline(source_rfm_file, source_model_file):
    """
    Calculate baseline performance on source domain
    (for comparison purposes)
    """
    print("\n📊 Calculating SOURCE baseline performance...")
    
    # Load source data and model
    source_rfm = pd.read_csv(source_rfm_file)
    
    with open(source_model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    source_model = model_data['model']
    source_scaler = model_data['scaler']
    source_labels = model_data['labels']
    
    # Prepare data
    X_source = source_rfm[['Recency', 'Frequency', 'Monetary']].dropna().values
    X_source_scaled = source_scaler.transform(X_source)
    
    # Evaluate
    metrics = {
        'silhouette': silhouette_score(X_source_scaled, source_labels),
        'davies_bouldin': davies_bouldin_score(X_source_scaled, source_labels),
        'calinski_harabasz': calinski_harabasz_score(X_source_scaled, source_labels),
        'n_clusters': len(np.unique(source_labels))
    }
    
    print(f"  ✓ Source Silhouette: {metrics['silhouette']:.3f}")
    print(f"  ✓ Source Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
    
    return metrics