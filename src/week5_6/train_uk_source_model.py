"""
Train RFM Customer Segmentation Model on UK Source Data (Experiment 5)
Uses the same methodology as week2/improved_train_all_domains.py
"""

import pandas as pd
import numpy as np
import sys
import os

# Add week2 to path to import the baseline models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'week2'))
from improved_baseline_models import RFMSegmentationModel, plot_elbow_curve

# Configuration
UK_SOURCE_CONFIG = {
    'name': 'UK Retail - Source Domain',
    'rfm_file': 'exp5_uk_source_RFM_FIXED.csv',
    'description': 'UK online retail customers (source domain for experiment 5)',
    'experiment': 'Experiment 5: UK (source) -> France (target)'
}

# Extended K range for better segmentation
K_RANGE = [3, 4, 5, 6, 7, 8]
RFM_FEATURES = ['Recency', 'Frequency', 'Monetary']


def train_uk_source_model():
    """
    Train RFM clustering model for UK source domain with improved analysis
    
    Returns:
    --------
    metrics : dict
        Performance metrics for this domain
    """
    print("\n" + "="*80)
    print(f"TRAINING: {UK_SOURCE_CONFIG['name']}")
    print(f"   Description: {UK_SOURCE_CONFIG['description']}")
    print(f"   Experiment: {UK_SOURCE_CONFIG['experiment']}")
    print("="*80)
    
    # Load RFM data
    print(f"\nLoading: {UK_SOURCE_CONFIG['rfm_file']}")
    df_rfm = pd.read_csv(UK_SOURCE_CONFIG['rfm_file'])
    
    # Rename CustomerID to customer_id for consistency with week2 code
    if 'CustomerID' in df_rfm.columns:
        df_rfm.rename(columns={'CustomerID': 'customer_id'}, inplace=True)
    
    print(f"Loaded {len(df_rfm)} customers")
    
    # Display RFM statistics
    print(f"\nRFM Statistics:")
    stats = df_rfm[RFM_FEATURES].describe()
    print(stats)
    
    # Additional statistics
    print(f"\nDistribution Insights:")
    print(f"   Recency Range: {df_rfm['Recency'].min():.0f} - {df_rfm['Recency'].max():.0f} days")
    print(f"   Frequency Range: {df_rfm['Frequency'].min():.0f} - {df_rfm['Frequency'].max():.0f} purchases")
    print(f"   Monetary Range: ${df_rfm['Monetary'].min():.2f} - ${df_rfm['Monetary'].max():.2f}")
    print(f"   Median Customer: {df_rfm['Recency'].median():.0f}d recency, "
          f"{df_rfm['Frequency'].median():.0f} purchases, ${df_rfm['Monetary'].median():.2f}")
    
    # Initialize model
    model = RFMSegmentationModel(rfm_features=RFM_FEATURES)
    
    # Prepare data
    X_scaled, customer_ids = model.prepare_data(df_rfm)
    df_clean = df_rfm[df_rfm['customer_id'].isin(customer_ids)]
    
    # Train K-Means with optimal k selection
    print(f"\nTraining K-Means on RFM features...")
    best_kmeans, results = model.train_kmeans(X_scaled, k_range=K_RANGE)
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    print("\nCreated output directories: models/, results/, plots/")
    
    # Plot elbow curve
    plot_elbow_curve(results, save_path='plots/exp5_uk_source_elbow_curve.png')
    
    # Evaluate model
    print(f"\nEvaluating model...")
    metrics = model.evaluate(best_kmeans, X_scaled, model.labels)
    
    print(f"\nCLUSTERING RESULTS:")
    print(f"   Optimal k: {model.best_k}")
    print(f"   Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"   Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
    print(f"   Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
    
    # Get segment profiles
    print(f"\nCreating segment profiles...")
    profiles = model.get_segment_profiles(df_clean)
    
    print("\n" + "="*80)
    print("SEGMENT PROFILES (Sorted by Customer Value):")
    print("="*80)
    
    # Calculate population means for comparison
    pop_r_mean = df_clean['Recency'].mean()
    pop_f_mean = df_clean['Frequency'].mean()
    pop_m_mean = df_clean['Monetary'].mean()
    
    for idx, row in profiles.iterrows():
        print(f"\n{row['Segment_Name']}")
        print(f"   Size: {int(row['Recency_count'])} customers ({row['Recency_count']/len(df_clean)*100:.1f}%)")
        print(f"   Value Score: {row['Value_Score']:.1f}/100")
        
        # Recency comparison
        r_diff = ((pop_r_mean - row['Recency_mean']) / pop_r_mean) * 100
        r_indicator = 'better' if r_diff > 0 else 'worse'
        print(f"   Recency:   {row['Recency_mean']:.1f} days (pop: {pop_r_mean:.1f}) {abs(r_diff):.0f}% {r_indicator}")
        
        # Frequency comparison
        f_diff = ((row['Frequency_mean'] - pop_f_mean) / pop_f_mean) * 100
        f_indicator = 'better' if f_diff > 0 else 'worse'
        print(f"   Frequency: {row['Frequency_mean']:.1f} purchases (pop: {pop_f_mean:.1f}) {abs(f_diff):.0f}% {f_indicator}")
        
        # Monetary comparison
        m_diff = ((row['Monetary_mean'] - pop_m_mean) / pop_m_mean) * 100
        m_indicator = 'better' if m_diff > 0 else 'worse'
        print(f"   Monetary:  ${row['Monetary_mean']:.2f} (pop: ${pop_m_mean:.2f}) {abs(m_diff):.0f}% {m_indicator}")
    
    # Distribution analysis
    print(f"\nCustomer Distribution Analysis:")
    total_customers = len(df_clean)
    for idx, row in profiles.iterrows():
        size = int(row['Recency_count'])
        pct = (size / total_customers) * 100
        print(f"   {row['Segment_Name']}: {size} ({pct:.1f}%) - Value Score: {row['Value_Score']:.1f}")
    
    # Save segment profiles
    profiles.to_csv('results/exp5_uk_source_segment_profiles.csv')
    print(f"\nSaved: results/exp5_uk_source_segment_profiles.csv")
    
    # Save customer segments
    df_segments = df_clean.copy()
    df_segments['Segment'] = model.labels
    df_segments['Segment_Name'] = df_segments['Segment'].map(
        lambda x: profiles.loc[x, 'Segment_Name']
    )
    df_segments['Value_Score'] = df_segments['Segment'].map(
        lambda x: profiles.loc[x, 'Value_Score']
    )
    df_segments.to_csv('results/exp5_uk_source_customer_segments.csv', index=False)
    print(f"Saved: results/exp5_uk_source_customer_segments.csv")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    
    # 3D RFM scatter plot
    model.plot_rfm_3d(df_clean, 
                      title=f"RFM Segments: {UK_SOURCE_CONFIG['name']}", 
                      save_path='plots/exp5_uk_source_rfm_3d.png')
    
    # Segment profiles
    model.plot_segment_profiles(profiles, 
                                title=f"Segment Profiles: {UK_SOURCE_CONFIG['name']}", 
                                save_path='plots/exp5_uk_source_segment_profiles.png')
    
    # Customer distribution
    model.plot_segment_distribution(profiles, 
                                   title=f"Customer Distribution: {UK_SOURCE_CONFIG['name']}", 
                                   save_path='plots/exp5_uk_source_distribution.png')
    
    # Save model
    model.save_model('models/exp5_uk_source_rfm_kmeans_model.pkl')
    
    # Calculate balance metrics
    segment_sizes = [int(row['Recency_count']) for _, row in profiles.iterrows()]
    balance_ratio = max(segment_sizes) / min(segment_sizes) if segment_sizes else 0
    
    # Create summary report
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    print(f"\nModel Performance:")
    print(f"   Customers Segmented: {len(df_clean):,}")
    print(f"   Optimal Number of Segments: {model.best_k}")
    print(f"   Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"   Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
    print(f"   Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
    
    print(f"\nSegment Balance:")
    print(f"   Largest Segment: {max(segment_sizes):,} customers "
          f"({max(segment_sizes)/len(df_clean)*100:.1f}%)")
    print(f"   Smallest Segment: {min(segment_sizes):,} customers "
          f"({min(segment_sizes)/len(df_clean)*100:.1f}%)")
    print(f"   Balance Ratio: {balance_ratio:.2f}")
    
    if balance_ratio <= 2.0:
        print(f"   Status: Well-balanced")
    elif balance_ratio <= 4.0:
        print(f"   Status: Acceptable")
    else:
        print(f"   Status: Imbalanced - consider different k")
    
    print(f"\nTop Segment:")
    top_segment = profiles.iloc[0]
    print(f"   Name: {top_segment['Segment_Name']}")
    print(f"   Value Score: {top_segment['Value_Score']:.1f}/100")
    print(f"   Size: {int(top_segment['Recency_count'])} customers")
    
    print(f"\nModel Quality Assessment:")
    if metrics['silhouette_score'] > 0.4 and metrics['davies_bouldin_index'] < 1.5:
        print("   Excellent segmentation quality")
        print("   High transfer learning potential")
    elif metrics['silhouette_score'] > 0.25 and metrics['davies_bouldin_index'] < 2.0:
        print("   Good segmentation quality")
        print("   Moderate transfer learning potential - may need fine-tuning")
    else:
        print("   Weak segmentation quality")
        print("   Low transfer learning potential - fresh model recommended")
    
    # Save summary metrics
    summary = {
        'experiment': UK_SOURCE_CONFIG['experiment'],
        'domain': UK_SOURCE_CONFIG['name'],
        'n_customers': len(df_clean),
        'optimal_k': model.best_k,
        'silhouette_score': metrics['silhouette_score'],
        'davies_bouldin_index': metrics['davies_bouldin_index'],
        'calinski_harabasz_score': metrics['calinski_harabasz_score'],
        'n_segments': metrics['n_clusters'],
        'largest_segment_pct': max(segment_sizes) / len(df_clean) * 100,
        'smallest_segment_pct': min(segment_sizes) / len(df_clean) * 100,
        'balance_ratio': balance_ratio,
        'top_segment': top_segment['Segment_Name'],
        'top_segment_value': top_segment['Value_Score']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('results/exp5_uk_source_training_summary.csv', index=False)
    print(f"\nSaved: results/exp5_uk_source_training_summary.csv")
    
    print("\n" + "="*80)
    print("DELIVERABLES CREATED:")
    print("="*80)
    print("  models/exp5_uk_source_rfm_kmeans_model.pkl - Trained model")
    print("  results/exp5_uk_source_segment_profiles.csv - Segment statistics")
    print("  results/exp5_uk_source_customer_segments.csv - Customer assignments")
    print("  results/exp5_uk_source_training_summary.csv - Performance metrics")
    print("  plots/exp5_uk_source_elbow_curve.png - K selection analysis")
    print("  plots/exp5_uk_source_rfm_3d.png - 3D RFM visualization")
    print("  plots/exp5_uk_source_segment_profiles.png - Segment profiles")
    print("  plots/exp5_uk_source_distribution.png - Customer distribution")
    
    print("\nNext Steps:")
    print("  1. Review segment profiles and customer assignments")
    print("  2. Validate segmentation quality with business experts")
    print("  3. Test transfer learning to France target domain")
    print("  4. Compare with France target segmentation")
    print("  5. Evaluate transferability metrics")
    
    return summary


def main():
    """
    Main training pipeline for UK source domain
    """
    print("\n" + "="*40)
    print("RFM CUSTOMER SEGMENTATION")
    print("UK RETAIL SOURCE DOMAIN TRAINING")
    print("="*40)
    print(f"\nRFM Features: {RFM_FEATURES}")
    print(f"K range tested: {K_RANGE}")
    print(f"Algorithm: K-Means with optimal k selection (Silhouette Score)")
    print(f"Methodology: Same as week2/improved_train_all_domains.py")
    
    # Change to week5_6 directory to save files there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"\nWorking directory: {os.getcwd()}")
    
    # Train model
    try:
        summary = train_uk_source_model()
        
        print("\n" + "="*40)
        print("TRAINING COMPLETE!")
        print("="*40)
        print(f"\nModel successfully trained on {summary['n_customers']:,} customers")
        print(f"Silhouette Score: {summary['silhouette_score']:.4f}")
        print(f"Optimal k: {summary['optimal_k']}")
        print(f"\nAll files saved in /src/week5_6/")
        
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {UK_SOURCE_CONFIG['rfm_file']}")
        print(f"Please ensure the file exists in the current directory")
        print(f"Error details: {str(e)}")
        return 1
    except Exception as e:
        print(f"\nERROR during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
