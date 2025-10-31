"""
Training Script for All 4 Domain Sources - RFM Customer Segmentation
Trains baseline K-Means models on RFM data and saves results
"""

import pandas as pd
import numpy as np
from baseline_models import RFMSegmentationModel, plot_elbow_curve
import os

# Configuration
DOMAINS = {
    'domain_pair1': {
        'name': 'Cleaning & Household',
        'rfm_file': 'domain_pair1_source_RFM.csv',
        'description': 'High transferability to Foodgrains, Oil & Masala'
    },
    'domain_pair2': {
        'name': 'Snacks & Branded Foods',
        'rfm_file': 'domain_pair2_source_RFM.csv',
        'description': 'Moderate transferability to Fruits & Vegetables'
    },
    'domain_pair3': {
        'name': 'Premium Segment',
        'rfm_file': 'domain_pair3_source_RFM.csv',
        'description': 'Low transferability - Price segments'
    },
    'domain_pair4': {
        'name': 'Popular Brands',
        'rfm_file': 'domain_pair4_source_RFM.csv',
        'description': 'Low-Moderate transferability - Brand popularity'
    }
}

K_RANGE = [3, 4, 5, 6, 7, 8]  # Range of k values to test
RFM_FEATURES = ['Recency', 'Frequency', 'Monetary']


def train_domain(domain_id, domain_info):
    """
    Train RFM clustering model for a single domain
    
    Parameters:
    -----------
    domain_id : str
        Domain identifier (e.g., 'domain_pair1')
    domain_info : dict
        Domain information including name and file path
        
    Returns:
    --------
    metrics : dict
        Performance metrics for this domain
    """
    print("\n" + "="*80)
    print(f"🎯 TRAINING: {domain_info['name']}")
    print(f"   Description: {domain_info['description']}")
    print("="*80)
    
    # Load RFM data
    print(f"\n📂 Loading: {domain_info['rfm_file']}")
    df_rfm = pd.read_csv(domain_info['rfm_file'])
    print(f"✓ Loaded {len(df_rfm)} customers")
    
    # Display RFM statistics
    print(f"\n📊 RFM Statistics:")
    print(df_rfm[RFM_FEATURES].describe())
    
    # Initialize model
    model = RFMSegmentationModel(rfm_features=RFM_FEATURES)
    
    # Prepare data
    X_scaled, customer_ids = model.prepare_data(df_rfm)
    df_clean = df_rfm[df_rfm['customer_id'].isin(customer_ids)]
    
    # Train K-Means with optimal k selection
    print(f"\n🔄 Training K-Means on RFM features...")
    best_kmeans, results = model.train_kmeans(X_scaled, k_range=K_RANGE)
    
    # Plot elbow curve
    plot_elbow_curve(results, save_path=f'plots/{domain_id}_elbow_curve.png')
    
    # Evaluate model
    print(f"\n📈 Evaluating model...")
    metrics = model.evaluate(best_kmeans, X_scaled, model.labels)
    
    print(f"\n✅ RESULTS:")
    print(f"   Optimal k: {model.best_k}")
    print(f"   Silhouette Score: {metrics['silhouette_score']:.3f}")
    print(f"   Davies-Bouldin Index: {metrics['davies_bouldin_index']:.3f}")
    print(f"   Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.1f}")
    print(f"\n   Customer Distribution:")
    total_customers = len(model.labels)
    for segment_id, size in metrics['cluster_sizes'].items():
        percentage = (size / total_customers) * 100
        print(f"      Segment {segment_id}: {size} customers ({percentage:.1f}%)")
    
    # Get segment profiles
    print(f"\n📊 Creating segment profiles...")
    profiles = model.get_segment_profiles(df_clean)
    print("\n" + "="*80)
    print("SEGMENT PROFILES (Compared to Overall Population):")
    print("="*80)
    
    # Calculate population means for comparison
    pop_r_mean = df_clean['Recency'].mean()
    pop_f_mean = df_clean['Frequency'].mean()
    pop_m_mean = df_clean['Monetary'].mean()
    
    for idx, row in profiles.iterrows():
        print(f"\n🔸 Segment {idx}: {row['Segment_Name']}")
        print(f"   Size: {int(row['Recency_count'])} customers ({row['Recency_count']/len(df_clean)*100:.1f}%)")
        print(f"   Recency:   {row['Recency_mean']:.1f} days  (pop avg: {pop_r_mean:.1f}) {'🟢 Better' if row['Recency_mean'] < pop_r_mean else '🔴 Worse'}")
        print(f"   Frequency: {row['Frequency_mean']:.1f} purchases (pop avg: {pop_f_mean:.1f}) {'🟢 Better' if row['Frequency_mean'] > pop_f_mean else '🔴 Worse'}")
        print(f"   Monetary:  ₹{row['Monetary_mean']:.2f} (pop avg: ₹{pop_m_mean:.2f}) {'🟢 Better' if row['Monetary_mean'] > pop_m_mean else '🔴 Worse'}")
    
    # Save segment profiles
    profiles.to_csv(f'results/{domain_id}_segment_profiles.csv')
    print(f"\n✓ Saved: results/{domain_id}_segment_profiles.csv")
    
    # Save customer segments
    df_segments = df_clean.copy()
    df_segments['Segment'] = model.labels
    df_segments['Segment_Name'] = df_segments['Segment'].map(
        lambda x: profiles.loc[x, 'Segment_Name']
    )
    df_segments.to_csv(f'results/{domain_id}_customer_segments.csv', index=False)
    print(f"✓ Saved: results/{domain_id}_customer_segments.csv")
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    
    # 3D RFM scatter plot
    model.plot_rfm_3d(df_clean, 
                      title=f"RFM Segments: {domain_info['name']}", 
                      save_path=f'plots/{domain_id}_rfm_3d.png')
    
    # Segment profiles
    model.plot_segment_profiles(profiles, 
                                title=f"Segment Profiles: {domain_info['name']}", 
                                save_path=f'plots/{domain_id}_segment_profiles.png')
    
    # Customer distribution
    model.plot_segment_distribution(profiles, 
                                   title=f"Customer Distribution: {domain_info['name']}", 
                                   save_path=f'plots/{domain_id}_distribution.png')
    
    # Save model
    model.save_model(f'models/{domain_id}_rfm_kmeans_model.pkl')
    
    # Return metrics for summary
    return {
        'domain_id': domain_id,
        'domain_name': domain_info['name'],
        'n_customers': len(df_clean),
        'optimal_k': model.best_k,
        'silhouette_score': metrics['silhouette_score'],
        'davies_bouldin_index': metrics['davies_bouldin_index'],
        'calinski_harabasz_score': metrics['calinski_harabasz_score'],
        'n_segments': metrics['n_clusters'],
        'largest_segment_pct': max(metrics['cluster_sizes'].values()) / len(model.labels) * 100,
        'smallest_segment_pct': min(metrics['cluster_sizes'].values()) / len(model.labels) * 100
    }


def create_comparison_report(all_results):
    """
    Create a comprehensive comparison report across all domains
    
    Parameters:
    -----------
    all_results : list
        List of result dictionaries from each domain
    """
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE ANALYSIS - CROSS-DOMAIN COMPARISON")
    print("="*80)
    
    df_summary = pd.DataFrame(all_results)
    
    # Display summary table
    print("\n1️⃣ PERFORMANCE SUMMARY:")
    print(df_summary.to_string(index=False))
    
    # Quality analysis
    print("\n2️⃣ CLUSTER QUALITY ANALYSIS:")
    print(f"   Best Silhouette Score: {df_summary['silhouette_score'].max():.3f} ({df_summary.loc[df_summary['silhouette_score'].idxmax(), 'domain_name']})")
    print(f"   Worst Silhouette Score: {df_summary['silhouette_score'].min():.3f} ({df_summary.loc[df_summary['silhouette_score'].idxmin(), 'domain_name']})")
    print(f"   Average Silhouette Score: {df_summary['silhouette_score'].mean():.3f}")
    
    # Segment distribution analysis
    print("\n3️⃣ SEGMENTATION CHARACTERISTICS:")
    print(f"   Most segments: {df_summary['optimal_k'].max()} ({df_summary.loc[df_summary['optimal_k'].idxmax(), 'domain_name']})")
    print(f"   Least segments: {df_summary['optimal_k'].min()} ({df_summary.loc[df_summary['optimal_k'].idxmin(), 'domain_name']})")
    print(f"   Average segments: {df_summary['optimal_k'].mean():.1f}")
    
    # Balance analysis
    print("\n4️⃣ SEGMENT BALANCE:")
    df_summary['balance_ratio'] = df_summary['largest_segment_pct'] / df_summary['smallest_segment_pct']
    print(f"   Most balanced: {df_summary.loc[df_summary['balance_ratio'].idxmin(), 'domain_name']} (ratio: {df_summary['balance_ratio'].min():.2f})")
    print(f"   Least balanced: {df_summary.loc[df_summary['balance_ratio'].idxmax(), 'domain_name']} (ratio: {df_summary['balance_ratio'].max():.2f})")
    
    # Transferability insights
    print("\n5️⃣ TRANSFERABILITY INSIGHTS:")
    transferability_map = {
        'domain_pair1': 'HIGH (0.903)',
        'domain_pair2': 'MODERATE (0.548)',
        'domain_pair3': 'LOW (0.715)',
        'domain_pair4': 'LOW-MODERATE (0.874)'
    }
    
    for _, row in df_summary.iterrows():
        domain_id = row['domain_id']
        silhouette = row['silhouette_score']
        transfer_score = transferability_map.get(domain_id, 'Unknown')
        
        print(f"\n   {row['domain_name']}:")
        print(f"      Transferability: {transfer_score}")
        print(f"      Cluster Quality: {silhouette:.3f}")
        print(f"      Segments Found: {row['optimal_k']}")
        
        # Hypothesis about transfer learning
        if silhouette > 0.4:
            print(f"      ✅ Hypothesis: Strong clustering suggests clear customer segments")
            print(f"         Transfer learning should preserve segment structure well")
        elif silhouette > 0.25:
            print(f"      ⚠️  Hypothesis: Moderate clustering quality")
            print(f"         Transfer learning may need fine-tuning for target domain")
        else:
            print(f"      ❌ Hypothesis: Weak clustering suggests overlapping segments")
            print(f"         Transfer learning may require significant adaptation")


def main():
    """
    Main training pipeline for all domains
    """
    print("\n" + "🚀"*40)
    print("RFM CUSTOMER SEGMENTATION - BASELINE CLUSTERING PIPELINE")
    print("🚀"*40)
    print(f"\nRFM Features: {RFM_FEATURES}")
    print(f"K range tested: {K_RANGE}")
    print(f"Domains to train: {len(DOMAINS)}")
    print(f"Algorithm: K-Means with optimal k selection (Silhouette Score)")
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    print("\n✓ Created output directories: models/, results/, plots/")
    
    # Train all domains
    all_results = []
    
    for domain_id, domain_info in DOMAINS.items():
        try:
            result = train_domain(domain_id, domain_info)
            all_results.append(result)
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: File not found for {domain_id}: {domain_info['rfm_file']}")
            print(f"   Please ensure the file exists in the current directory")
            continue
        except Exception as e:
            print(f"\n❌ ERROR training {domain_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comprehensive analysis
    if all_results:
        create_comparison_report(all_results)
        
        # Save summary
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv('results/baseline_performance.csv', index=False)
        print(f"\n✅ Saved summary: results/baseline_performance.csv")
    
    # Final summary
    print("\n" + "🎉"*40)
    print("TRAINING COMPLETE!")
    print("🎉"*40)
    print(f"\n📦 Deliverables Created:")
    print(f"   ✅ baseline_models.py (RFM clustering implementation)")
    print(f"   ✅ {len(all_results)} trained models (.pkl files)")
    print(f"   ✅ baseline_performance.csv (performance metrics)")
    print(f"   ✅ {len(all_results)} customer segment CSVs")
    print(f"   ✅ {len(all_results)} segment profile CSVs")
    print(f"   ✅ {len(all_results) * 4} visualizations")
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Customers Segmented: {summary_df['n_customers'].sum():,}")
        print(f"   Average Silhouette Score: {summary_df['silhouette_score'].mean():.3f}")
        print(f"   Average Segments per Domain: {summary_df['optimal_k'].mean():.1f}")
    
    print("\n✅ All files saved in:")
    print(f"   📁 models/       - Trained RFM clustering models")
    print(f"   📁 results/      - Performance metrics, segment profiles, customer assignments")
    print(f"   📁 plots/        - All visualizations (elbow, 3D RFM, profiles, distribution)")
    
    print("\n🎯 Next Steps:")
    print("   1. Review segment profiles and customer distributions")
    print("   2. Analyze baseline_performance.csv")
    print("   3. Examine RFM 3D plots for segment separation")
    print("   4. Test transfer learning: Apply source models to target RFM data")
    print("   5. Compare source vs target segment quality")
    print("   6. Validate Week 1 transferability predictions")


if __name__ == "__main__":
    main()