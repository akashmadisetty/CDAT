"""
Test BASELINE ONLY: Pair 1 Gold Source -> Silver Target
Direct transfer without any fine-tuning
Should perform poorly since CLI says transferability is very low (0.22)
"""

import pandas as pd
import numpy as np
import sys
import os
import pickle
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Add week2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'week2'))
from improved_baseline_models import RFMSegmentationModel

# Configuration
RFM_FEATURES = ['Recency', 'Frequency', 'Monetary']
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def load_trained_model(model_path):
    """Load the trained Gold source model"""
    print(f"\nLoading trained Gold source model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Loaded model with k={model_data['best_k']} clusters")
    return model_data


def evaluate_clustering(X, labels, model_name):
    """
    Evaluate clustering performance
    
    Parameters:
    -----------
    X : numpy.ndarray
        Scaled feature matrix
    labels : numpy.ndarray
        Cluster labels
    model_name : str
        Name of the model being evaluated
        
    Returns:
    --------
    metrics : dict
        Performance metrics
    """
    # Remove noise points if any
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    
    if len(set(labels_filtered)) < 2:
        print(f"\n{model_name}: Not enough clusters to evaluate")
        return None
    
    metrics = {
        'model': model_name,
        'n_samples': len(X_filtered),
        'n_clusters': len(set(labels_filtered)),
        'silhouette_score': silhouette_score(X_filtered, labels_filtered),
        'davies_bouldin_index': davies_bouldin_score(X_filtered, labels_filtered),
        'calinski_harabasz_score': calinski_harabasz_score(X_filtered, labels_filtered)
    }
    
    # Cluster size distribution
    unique, counts = np.unique(labels_filtered, return_counts=True)
    metrics['cluster_sizes'] = dict(zip(unique, counts))
    metrics['largest_cluster_pct'] = max(counts) / len(labels_filtered) * 100
    metrics['smallest_cluster_pct'] = min(counts) / len(labels_filtered) * 100
    
    return metrics


def test_baseline_transfer(source_model_data, df_test):
    """
    Test direct transfer without fine-tuning (BASELINE ONLY)
    
    Parameters:
    -----------
    source_model_data : dict
        Loaded source model data
    df_test : pandas.DataFrame
        Test data
        
    Returns:
    --------
    metrics : dict
        Performance metrics
    labels : numpy.ndarray
        Predicted cluster labels
    """
    print("\n" + "="*80)
    print("BASELINE TEST: DIRECT TRANSFER (No Fine-tuning)")
    print("="*80)
    print("\nApplying Gold source model directly to ALL Silver data...")
    print("(CLI predicted this would fail - transferability score: 0.22)")
    
    # Extract test features
    X_test = df_test[RFM_FEATURES].values
    
    # Scale using source scaler
    X_test_scaled = source_model_data['scaler'].transform(X_test)
    
    # Predict using source model
    labels = source_model_data['model'].predict(X_test_scaled)
    
    print(f"\nPredicted {len(set(labels))} clusters on {len(df_test)} test customers")
    print(f"(Gold source model has {source_model_data['best_k']} clusters)")
    
    # Evaluate
    metrics = evaluate_clustering(X_test_scaled, labels, "Baseline Direct Transfer")
    
    if metrics:
        print(f"\nBaseline Direct Transfer Performance:")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
        print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
        print(f"  Number of Clusters: {metrics['n_clusters']}")
        
        print(f"\nCluster Distribution:")
        for cluster, size in metrics['cluster_sizes'].items():
            pct = (size / metrics['n_samples']) * 100
            print(f"    Cluster {cluster}: {size} customers ({pct:.1f}%)")
        
        # Assess quality
        print("\n" + "="*80)
        print("QUALITY ASSESSMENT:")
        print("="*80)
        
        sil = metrics['silhouette_score']
        db = metrics['davies_bouldin_index']
        
        print(f"\nSilhouette Score: {sil:.4f}")
        if sil > 0.5:
            print("  -> EXCELLENT (>0.5) - Clusters are well-separated")
        elif sil > 0.35:
            print("  -> GOOD (>0.35) - Clear cluster structure")
        elif sil > 0.25:
            print("  -> ACCEPTABLE (>0.25) - Moderate separation")
        else:
            print("  -> POOR (<0.25) - Weak or overlapping clusters")
        
        print(f"\nDavies-Bouldin Index: {db:.4f}")
        if db < 1.0:
            print("  -> EXCELLENT (<1.0) - Clusters are distinct")
        elif db < 1.5:
            print("  -> GOOD (<1.5) - Well-separated clusters")
        elif db < 2.0:
            print("  -> ACCEPTABLE (<2.0) - Some overlap")
        else:
            print("  -> POOR (>2.0) - Clusters too similar")
        
        # Missing clusters analysis
        missing_clusters = source_model_data['best_k'] - metrics['n_clusters']
        if missing_clusters > 0:
            print(f"\nWARNING: {missing_clusters} out of {source_model_data['best_k']} clusters are missing!")
            print("  -> Silver customers don't span the full range of Gold customer types")
            print("  -> This confirms domains are very different")
        
        # Overall verdict
        print("\n" + "="*80)
        print("VERDICT:")
        print("="*80)
        
        if sil < 0.35 or db > 1.5 or missing_clusters > source_model_data['best_k'] / 2:
            print("\nBaseline transfer FAILED (as predicted by CLI)")
            print("Reasons:")
            if sil < 0.35:
                print("  - Poor silhouette score (weak cluster separation)")
            if db > 1.5:
                print("  - High Davies-Bouldin index (overlapping clusters)")
            if missing_clusters > 0:
                print(f"  - Missing {missing_clusters} clusters (incomplete coverage)")
            print("\nCLI was CORRECT: Transferability score 0.22 = Don't transfer")
        else:
            print("\nBaseline transfer performed better than expected")
            print("This contradicts CLI's low transferability score (0.22)")
    
    return metrics, labels


def save_results(metrics, df_test, labels):
    """Save test results to CSV"""
    
    if metrics:
        # Save metrics
        metrics_df = pd.DataFrame([{
            'model': 'Baseline Direct Transfer',
            'transferability_score': 0.2218,
            'n_samples': metrics['n_samples'],
            'n_clusters': metrics['n_clusters'],
            'source_k': 8,
            'missing_clusters': 8 - metrics['n_clusters'],
            'silhouette_score': metrics['silhouette_score'],
            'davies_bouldin_index': metrics['davies_bouldin_index'],
            'calinski_harabasz_score': metrics['calinski_harabasz_score'],
            'largest_cluster_pct': metrics['largest_cluster_pct'],
            'smallest_cluster_pct': metrics['smallest_cluster_pct']
        }])
        metrics_df.to_csv('results/pair1_baseline_only_metrics.csv', index=False)
        print(f"\nSaved: results/pair1_baseline_only_metrics.csv")
    
    # Save predictions
    df_predictions = df_test.copy()
    df_predictions['baseline_cluster'] = labels
    df_predictions.to_csv('results/pair1_baseline_only_predictions.csv', index=False)
    print(f"Saved: results/pair1_baseline_only_predictions.csv")


def main():
    """
    Main pipeline for testing baseline transfer
    """
    print("\n" + "="*80)
    print("BASELINE TRANSFER TEST: Pair 1 Gold (Source) -> Silver (Target)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Transfer Method: Direct (no fine-tuning)")
    print(f"  RFM Features: {RFM_FEATURES}")
    print(f"  CLI Transferability Score: 0.2218 (VERY LOW)")
    print(f"  CLI Recommendation: Train from scratch (don't transfer)")
    
    # Change to week5_6 directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    
    # Load trained Gold source model
    model_path = 'models/pair1_gold_source_rfm_kmeans_model.pkl'
    source_model_data = load_trained_model(model_path)
    
    # Load ALL Silver target data (no split - test on everything)
    print(f"\nLoading ALL Silver target RFM data...")
    df_silver = pd.read_csv('../../data/processed/ecommerce/pair1_silver_target_RFM.csv')
    
    print(f"Loaded {len(df_silver)} Silver customers (testing on all of them)")
    
    # Test baseline transfer
    metrics, labels = test_baseline_transfer(source_model_data, df_silver)
    
    # Save results
    save_results(metrics, df_silver, labels)
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print("\nDeliverables:")
    print("  results/pair1_baseline_only_metrics.csv - Performance metrics")
    print("  results/pair1_baseline_only_predictions.csv - All Silver predictions")


if __name__ == "__main__":
    main()
