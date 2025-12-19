"""
Test Transfer Learning: UK Source -> France Target
Tests both direct transfer and fine-tuned transfer based on transferability analysis
Recommendation: Fine-tune with 15% of target data
"""

import pandas as pd
import numpy as np
import sys
import os
import pickle
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Add week2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'week2'))
from improved_baseline_models import RFMSegmentationModel

# Configuration
FINE_TUNE_PERCENTAGE = 0.15  # 15% as recommended by transferability analysis
RFM_FEATURES = ['Recency', 'Frequency', 'Monetary']
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def load_trained_model(model_path):
    """Load the trained UK source model"""
    print(f"\nLoading trained UK source model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Loaded model with k={model_data['best_k']} clusters")
    return model_data


def split_target_data(df, fine_tune_pct=0.15):
    """
    Split target data into fine-tuning and test sets
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Target domain RFM data
    fine_tune_pct : float
        Percentage of data for fine-tuning (0.15 = 15%)
        
    Returns:
    --------
    df_finetune : pandas.DataFrame
        Fine-tuning subset
    df_test : pandas.DataFrame
        Test subset
    """
    n_total = len(df)
    n_finetune = int(n_total * fine_tune_pct)
    
    # Shuffle and split
    df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df_finetune = df_shuffled.iloc[:n_finetune].copy()
    df_test = df_shuffled.iloc[n_finetune:].copy()
    
    print(f"\nTarget Data Split:")
    print(f"  Total: {n_total} customers")
    print(f"  Fine-tuning: {len(df_finetune)} customers ({fine_tune_pct*100:.0f}%)")
    print(f"  Test: {len(df_test)} customers ({(1-fine_tune_pct)*100:.0f}%)")
    
    return df_finetune, df_test


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


def test_direct_transfer(source_model_data, df_test):
    """
    Test direct transfer without fine-tuning
    
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
    print("TEST 1: DIRECT TRANSFER (No Fine-tuning)")
    print("="*80)
    print("\nApplying UK source model directly to France test data...")
    
    # Extract test features
    X_test = df_test[RFM_FEATURES].values
    
    # Scale using source scaler
    X_test_scaled = source_model_data['scaler'].transform(X_test)
    
    # Predict using source model
    labels = source_model_data['model'].predict(X_test_scaled)
    
    print(f"\nPredicted {len(set(labels))} clusters on {len(df_test)} test customers")
    
    # Evaluate
    metrics = evaluate_clustering(X_test_scaled, labels, "Direct Transfer")
    
    if metrics:
        print(f"\nDirect Transfer Performance:")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
        print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
        print(f"  Number of Clusters: {metrics['n_clusters']}")
        print(f"\nCluster Distribution:")
        for cluster, size in metrics['cluster_sizes'].items():
            pct = (size / metrics['n_samples']) * 100
            print(f"    Cluster {cluster}: {size} customers ({pct:.1f}%)")
    
    return metrics, labels


def test_finetuned_transfer(source_model_data, df_finetune, df_test):
    """
    Test transfer with fine-tuning
    
    Parameters:
    -----------
    source_model_data : dict
        Loaded source model data
    df_finetune : pandas.DataFrame
        Fine-tuning data
    df_test : pandas.DataFrame
        Test data
        
    Returns:
    --------
    metrics : dict
        Performance metrics
    labels : numpy.ndarray
        Predicted cluster labels
    finetuned_model : KMeans
        Fine-tuned model
    """
    print("\n" + "="*80)
    print("TEST 2: FINE-TUNED TRANSFER (15% Fine-tuning)")
    print("="*80)
    print("\nFine-tuning UK source model with France data...")
    
    # Extract fine-tuning features
    X_finetune = df_finetune[RFM_FEATURES].values
    
    # Scale using source scaler
    X_finetune_scaled = source_model_data['scaler'].transform(X_finetune)
    
    # Get source model's k
    k = source_model_data['best_k']
    
    # Initialize new K-Means with source centroids
    finetuned_model = KMeans(
        n_clusters=k,
        init=source_model_data['model'].cluster_centers_,  # Start from source centroids
        n_init=1,
        random_state=RANDOM_SEED
    )
    
    # Fine-tune on target data
    print(f"\nFine-tuning with {len(df_finetune)} France customers...")
    finetuned_model.fit(X_finetune_scaled)
    
    # Test on held-out data
    X_test = df_test[RFM_FEATURES].values
    X_test_scaled = source_model_data['scaler'].transform(X_test)
    labels = finetuned_model.predict(X_test_scaled)
    
    print(f"Predicted {len(set(labels))} clusters on {len(df_test)} test customers")
    
    # Evaluate
    metrics = evaluate_clustering(X_test_scaled, labels, "Fine-tuned Transfer")
    
    if metrics:
        print(f"\nFine-tuned Transfer Performance:")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
        print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
        print(f"  Number of Clusters: {metrics['n_clusters']}")
        print(f"\nCluster Distribution:")
        for cluster, size in metrics['cluster_sizes'].items():
            pct = (size / metrics['n_samples']) * 100
            print(f"    Cluster {cluster}: {size} customers ({pct:.1f}%)")
    
    return metrics, labels, finetuned_model


def compare_results(direct_metrics, finetuned_metrics):
    """
    Compare direct transfer vs fine-tuned transfer
    
    Parameters:
    -----------
    direct_metrics : dict
        Metrics from direct transfer
    finetuned_metrics : dict
        Metrics from fine-tuned transfer
    """
    print("\n" + "="*80)
    print("COMPARISON: Direct Transfer vs Fine-tuned Transfer")
    print("="*80)
    
    if not direct_metrics or not finetuned_metrics:
        print("\nCannot compare - one or both evaluations failed")
        return
    
    print("\n{:<30} {:<20} {:<20} {:<15}".format(
        "Metric", "Direct Transfer", "Fine-tuned", "Improvement"
    ))
    print("-" * 85)
    
    # Silhouette Score (higher is better)
    sil_direct = direct_metrics['silhouette_score']
    sil_finetuned = finetuned_metrics['silhouette_score']
    sil_improvement = ((sil_finetuned - sil_direct) / abs(sil_direct)) * 100 if sil_direct != 0 else 0
    print("{:<30} {:<20.4f} {:<20.4f} {:>+14.1f}%".format(
        "Silhouette Score", sil_direct, sil_finetuned, sil_improvement
    ))
    
    # Davies-Bouldin (lower is better)
    db_direct = direct_metrics['davies_bouldin_index']
    db_finetuned = finetuned_metrics['davies_bouldin_index']
    db_improvement = ((db_direct - db_finetuned) / db_direct) * 100 if db_direct != 0 else 0
    print("{:<30} {:<20.4f} {:<20.4f} {:>+14.1f}%".format(
        "Davies-Bouldin Index", db_direct, db_finetuned, db_improvement
    ))
    
    # Calinski-Harabasz (higher is better)
    ch_direct = direct_metrics['calinski_harabasz_score']
    ch_finetuned = finetuned_metrics['calinski_harabasz_score']
    ch_improvement = ((ch_finetuned - ch_direct) / ch_direct) * 100 if ch_direct != 0 else 0
    print("{:<30} {:<20.2f} {:<20.2f} {:>+14.1f}%".format(
        "Calinski-Harabasz Score", ch_direct, ch_finetuned, ch_improvement
    ))
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    # Determine winner
    better_count = 0
    if sil_improvement > 0:
        better_count += 1
    if db_improvement > 0:  # For DB, positive improvement means lower is better
        better_count += 1
    if ch_improvement > 0:
        better_count += 1
    
    if better_count >= 2:
        print("\nFine-tuning IMPROVED the model!")
        print(f"  {better_count}/3 metrics improved with 15% fine-tuning data")
        print("\nRecommendation: Use fine-tuned model for France target domain")
    elif better_count == 0:
        print("\nDirect transfer performed BETTER!")
        print("  Fine-tuning did not improve performance")
        print("\nRecommendation: Use direct transfer without fine-tuning")
    else:
        print("\nResults are MIXED!")
        print(f"  {better_count}/3 metrics improved")
        print("\nRecommendation: Evaluate based on business priorities")
    
    print("\nKey Insights:")
    if abs(sil_improvement) < 5 and abs(db_improvement) < 5 and abs(ch_improvement) < 5:
        print("  - Performance differences are minimal (<5%)")
        print("  - Both approaches are viable")
    else:
        print(f"  - Largest improvement: {max(abs(sil_improvement), abs(db_improvement), abs(ch_improvement)):.1f}%")
    
    print(f"  - Test sample size: {direct_metrics['n_samples']} customers")
    print(f"  - Number of clusters: {direct_metrics['n_clusters']}")


def save_results(direct_metrics, finetuned_metrics, df_test, direct_labels, finetuned_labels):
    """Save test results to CSV"""
    
    # Save metrics comparison
    if direct_metrics and finetuned_metrics:
        comparison = pd.DataFrame([
            {
                'metric': 'Silhouette Score',
                'direct_transfer': direct_metrics['silhouette_score'],
                'finetuned_transfer': finetuned_metrics['silhouette_score'],
                'improvement_pct': ((finetuned_metrics['silhouette_score'] - direct_metrics['silhouette_score']) / abs(direct_metrics['silhouette_score'])) * 100
            },
            {
                'metric': 'Davies-Bouldin Index',
                'direct_transfer': direct_metrics['davies_bouldin_index'],
                'finetuned_transfer': finetuned_metrics['davies_bouldin_index'],
                'improvement_pct': ((direct_metrics['davies_bouldin_index'] - finetuned_metrics['davies_bouldin_index']) / direct_metrics['davies_bouldin_index']) * 100
            },
            {
                'metric': 'Calinski-Harabasz Score',
                'direct_transfer': direct_metrics['calinski_harabasz_score'],
                'finetuned_transfer': finetuned_metrics['calinski_harabasz_score'],
                'improvement_pct': ((finetuned_metrics['calinski_harabasz_score'] - direct_metrics['calinski_harabasz_score']) / direct_metrics['calinski_harabasz_score']) * 100
            }
        ])
        comparison.to_csv('results/exp5_transfer_comparison.csv', index=False)
        print(f"\nSaved: results/exp5_transfer_comparison.csv")
    
    # Save predictions
    df_predictions = df_test.copy()
    df_predictions['direct_transfer_cluster'] = direct_labels
    df_predictions['finetuned_transfer_cluster'] = finetuned_labels
    df_predictions.to_csv('results/exp5_test_predictions.csv', index=False)
    print(f"Saved: results/exp5_test_predictions.csv")


def main():
    """
    Main pipeline for testing transfer learning
    """
    print("\n" + "="*80)
    print("TRANSFER LEARNING TEST: UK (Source) -> France (Target)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Fine-tuning percentage: {FINE_TUNE_PERCENTAGE*100:.0f}% (as recommended)")
    print(f"  RFM Features: {RFM_FEATURES}")
    print(f"  Random Seed: {RANDOM_SEED}")
    
    # Change to week5_6 directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    
    # Load trained UK source model
    model_path = 'models/exp5_uk_source_rfm_kmeans_model.pkl'
    source_model_data = load_trained_model(model_path)
    
    # Load France target data
    print(f"\nLoading France target RFM data...")
    df_france = pd.read_csv('exp5_france_target_RFM_FIXED.csv')
    
    if 'CustomerID' in df_france.columns:
        df_france.rename(columns={'CustomerID': 'customer_id'}, inplace=True)
    
    print(f"Loaded {len(df_france)} France customers")
    
    # Split target data
    df_finetune, df_test = split_target_data(df_france, FINE_TUNE_PERCENTAGE)
    
    # Test 1: Direct Transfer
    direct_metrics, direct_labels = test_direct_transfer(source_model_data, df_test)
    
    # Test 2: Fine-tuned Transfer
    finetuned_metrics, finetuned_labels, finetuned_model = test_finetuned_transfer(
        source_model_data, df_finetune, df_test
    )
    
    # Compare results
    compare_results(direct_metrics, finetuned_metrics)
    
    # Save results
    save_results(direct_metrics, finetuned_metrics, df_test, direct_labels, finetuned_labels)
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print("\nDeliverables:")
    print("  results/exp5_transfer_comparison.csv - Performance comparison")
    print("  results/exp5_test_predictions.csv - Test set predictions")


if __name__ == "__main__":
    main()
