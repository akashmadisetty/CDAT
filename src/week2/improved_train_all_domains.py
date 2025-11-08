"""
Improved Training Script for All 7 Domain Sources - RFM Customer Segmentation
Trains baseline K-Means models with better evaluation and segment naming
"""

import pandas as pd
import numpy as np
from improved_baseline_models import RFMSegmentationModel, plot_elbow_curve
import os

# Configuration
DOMAINS = {
    'domain_pair1': {
        'name': 'Cleaning & Household → Foodgrains',
        'rfm_file': 'domain_pair1_source_RFM.csv',
        'description': 'Cross-category transfer',
        'transferability': 'No Finetune'
    },
    'domain_pair2': {
        'name': 'Snacks → Garden, Kitchen',
        'rfm_file': 'domain_pair2_source_RFM.csv',
        'description': 'Partial transferability',
        'transferability': 'Partial'
    },
    'domain_pair3': {
        'name': 'Premium → Budget',
        'rfm_file': 'domain_pair3_source_RFM.csv',
        'description': 'Low transferability - Price segments',
        'transferability': 'Partial (Low)'
    },
    'domain_pair4': {
        'name': 'Premium → Mass-Market Beauty',
        'rfm_file': 'domain_pair4_source_RFM.csv',
        'description': 'Partial transferability - Beauty products',
        'transferability': 'Partial'
    },
    'domain_pair5': {
        'name': 'Eggs, Meat & Fish → Baby Care',
        'rfm_file': 'domain_pair5_source_RFM.csv',
        'description': 'New model required',
        'transferability': 'New Model'
    },
    'domain_pair6': {
        'name': 'Baby Care → Bakery, Cakes & Dairy',
        'rfm_file': 'domain_pair6_source_RFM.csv',
        'description': 'New model required',
        'transferability': 'New Model'
    },
    'domain_pair7': {
        'name': 'Beverages → Gourmet & World Food',
        'rfm_file': 'domain_pair7_source_RFM.csv',
        'description': 'Direct transfer without finetuning',
        'transferability': 'No Finetune'
    },
}

# Extended K range for better segmentation
K_RANGE = [3, 4, 5, 6, 7, 8]
RFM_FEATURES = ['Recency', 'Frequency', 'Monetary']


def train_domain(domain_id, domain_info):
    """
    Train RFM clustering model for a single domain with improved analysis
    
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
    print(f"   Transferability: {domain_info['transferability']}")
    print("="*80)
    
    # Load RFM data
    print(f"\n📂 Loading: {domain_info['rfm_file']}")
    df_rfm = pd.read_csv(domain_info['rfm_file'])
    print(f"✓ Loaded {len(df_rfm)} customers")
    
    # Display RFM statistics
    print(f"\n📊 RFM Statistics:")
    stats = df_rfm[RFM_FEATURES].describe()
    print(stats)
    
    # Additional statistics
    print(f"\n📈 Distribution Insights:")
    print(f"   Recency Range: {df_rfm['Recency'].min():.0f} - {df_rfm['Recency'].max():.0f} days")
    print(f"   Frequency Range: {df_rfm['Frequency'].min():.0f} - {df_rfm['Frequency'].max():.0f} purchases")
    print(f"   Monetary Range: ₹{df_rfm['Monetary'].min():.2f} - ₹{df_rfm['Monetary'].max():.2f}")
    print(f"   Median Customer: {df_rfm['Recency'].median():.0f}d recency, "
          f"{df_rfm['Frequency'].median():.0f} purchases, ₹{df_rfm['Monetary'].median():.2f}")
    
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
    
    print(f"\n✅ CLUSTERING RESULTS:")
    print(f"   Optimal k: {model.best_k}")
    print(f"   Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"   Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
    print(f"   Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
    
    # Get segment profiles
    print(f"\n📊 Creating segment profiles...")
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
        r_emoji = '🟢' if r_diff > 0 else '🔴'
        print(f"   Recency:   {row['Recency_mean']:.1f} days (pop: {pop_r_mean:.1f}) {r_emoji} {abs(r_diff):.0f}% {'better' if r_diff > 0 else 'worse'}")
        
        # Frequency comparison
        f_diff = ((row['Frequency_mean'] - pop_f_mean) / pop_f_mean) * 100
        f_emoji = '🟢' if f_diff > 0 else '🔴'
        print(f"   Frequency: {row['Frequency_mean']:.1f} purchases (pop: {pop_f_mean:.1f}) {f_emoji} {abs(f_diff):.0f}% {'better' if f_diff > 0 else 'worse'}")
        
        # Monetary comparison
        m_diff = ((row['Monetary_mean'] - pop_m_mean) / pop_m_mean) * 100
        m_emoji = '🟢' if m_diff > 0 else '🔴'
        print(f"   Monetary:  ₹{row['Monetary_mean']:.2f} (pop: ₹{pop_m_mean:.2f}) {m_emoji} {abs(m_diff):.0f}% {'better' if m_diff > 0 else 'worse'}")
    
    # Distribution analysis
    print(f"\n📊 Customer Distribution Analysis:")
    total_customers = len(df_clean)
    for idx, row in profiles.iterrows():
        size = int(row['Recency_count'])
        pct = (size / total_customers) * 100
        print(f"   {row['Segment_Name']}: {size} ({pct:.1f}%) - Value Score: {row['Value_Score']:.1f}")
    
    # Save segment profiles
    profiles.to_csv(f'results/{domain_id}_segment_profiles.csv')
    print(f"\n✓ Saved: results/{domain_id}_segment_profiles.csv")
    
    # Save customer segments
    df_segments = df_clean.copy()
    df_segments['Segment'] = model.labels
    df_segments['Segment_Name'] = df_segments['Segment'].map(
        lambda x: profiles.loc[x, 'Segment_Name']
    )
    df_segments['Value_Score'] = df_segments['Segment'].map(
        lambda x: profiles.loc[x, 'Value_Score']
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
    
    # Calculate balance metrics
    segment_sizes = [int(row['Recency_count']) for _, row in profiles.iterrows()]
    balance_ratio = max(segment_sizes) / min(segment_sizes) if segment_sizes else 0
    
    # Return metrics for summary
    return {
        'domain_id': domain_id,
        'domain_name': domain_info['name'],
        'transferability': domain_info['transferability'],
        'n_customers': len(df_clean),
        'optimal_k': model.best_k,
        'silhouette_score': metrics['silhouette_score'],
        'davies_bouldin_index': metrics['davies_bouldin_index'],
        'calinski_harabasz_score': metrics['calinski_harabasz_score'],
        'n_segments': metrics['n_clusters'],
        'largest_segment_pct': max(metrics['cluster_sizes'].values()) / len(model.labels) * 100,
        'smallest_segment_pct': min(metrics['cluster_sizes'].values()) / len(model.labels) * 100,
        'balance_ratio': balance_ratio,
        'top_segment': profiles.iloc[0]['Segment_Name'],
        'top_segment_value': profiles.iloc[0]['Value_Score']
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
    display_cols = ['domain_name', 'transferability', 'n_customers', 'optimal_k', 
                    'silhouette_score', 'davies_bouldin_index', 'calinski_harabasz_score']
    print(df_summary[display_cols].to_string(index=False))
    
    # Quality analysis
    print("\n2️⃣ CLUSTER QUALITY ANALYSIS:")
    print(f"\n   Silhouette Score Analysis:")
    best_sil = df_summary.loc[df_summary['silhouette_score'].idxmax()]
    worst_sil = df_summary.loc[df_summary['silhouette_score'].idxmin()]
    print(f"   ✅ Best: {best_sil['domain_name']} ({best_sil['silhouette_score']:.3f})")
    print(f"   ❌ Worst: {worst_sil['domain_name']} ({worst_sil['silhouette_score']:.3f})")
    print(f"   📊 Average: {df_summary['silhouette_score'].mean():.3f}")
    
    print(f"\n   Davies-Bouldin Index Analysis:")
    best_db = df_summary.loc[df_summary['davies_bouldin_index'].idxmin()]
    worst_db = df_summary.loc[df_summary['davies_bouldin_index'].idxmax()]
    print(f"   ✅ Best: {best_db['domain_name']} ({best_db['davies_bouldin_index']:.3f})")
    print(f"   ❌ Worst: {worst_db['domain_name']} ({worst_db['davies_bouldin_index']:.3f})")
    print(f"   📊 Average: {df_summary['davies_bouldin_index'].mean():.3f}")
    
    # Segmentation characteristics
    print("\n3️⃣ SEGMENTATION CHARACTERISTICS:")
    most_segs = df_summary.loc[df_summary['optimal_k'].idxmax()]
    least_segs = df_summary.loc[df_summary['optimal_k'].idxmin()]
    print(f"   Most segments: {int(most_segs['optimal_k'])} ({most_segs['domain_name']})")
    print(f"   Least segments: {int(least_segs['optimal_k'])} ({least_segs['domain_name']})")
    print(f"   Average segments: {df_summary['optimal_k'].mean():.1f}")
    
    # Balance analysis
    print("\n4️⃣ SEGMENT BALANCE ANALYSIS:")
    most_balanced = df_summary.loc[df_summary['balance_ratio'].idxmin()]
    least_balanced = df_summary.loc[df_summary['balance_ratio'].idxmax()]
    print(f"   Most balanced: {most_balanced['domain_name']} (ratio: {most_balanced['balance_ratio']:.2f})")
    print(f"   Least balanced: {least_balanced['domain_name']} (ratio: {least_balanced['balance_ratio']:.2f})")
    print(f"\n   Balance ratio interpretation:")
    print(f"   • 1.0-2.0: Well-balanced ✅")
    print(f"   • 2.0-4.0: Acceptable ⚠️")
    print(f"   • >4.0: Imbalanced ❌")
    
    # Transfer learning analysis
    print("\n5️⃣ TRANSFER LEARNING POTENTIAL ANALYSIS:")
    
    # Group by transferability
    for trans_type in df_summary['transferability'].unique():
        print(f"\n   📌 {trans_type} Domains:")
        subset = df_summary[df_summary['transferability'] == trans_type]
        for _, row in subset.iterrows():
            print(f"\n      {row['domain_name']}:")
            print(f"         Silhouette: {row['silhouette_score']:.3f}")
            print(f"         Davies-Bouldin: {row['davies_bouldin_index']:.3f}")
            print(f"         Segments: {int(row['optimal_k'])}")
            print(f"         Top Segment: {row['top_segment']} (Value: {row['top_segment_value']:.1f})")
            
            # Transfer learning hypothesis
            if row['silhouette_score'] > 0.4:
                print(f"         ✅ Strong clustering → High transfer potential")
            elif row['silhouette_score'] > 0.25:
                print(f"         ⚠️  Moderate clustering → May need fine-tuning")
            else:
                print(f"         ❌ Weak clustering → Transfer likely to fail")
    
    # Recommendations
    print("\n6️⃣ RECOMMENDATIONS BY DOMAIN:")
    for _, row in df_summary.iterrows():
        print(f"\n   {row['domain_name']}:")
        
        # Quality recommendation
        if row['silhouette_score'] > 0.4 and row['davies_bouldin_index'] < 1.5:
            print(f"      ✅ Excellent segmentation quality")
            print(f"      → Recommended: Use for transfer learning")
        elif row['silhouette_score'] > 0.25 and row['davies_bouldin_index'] < 2.0:
            print(f"      ⚠️  Good segmentation quality")
            print(f"      → Recommended: Test transfer, expect some fine-tuning")
        else:
            print(f"      ❌ Weak segmentation quality")
            print(f"      → Recommended: Train fresh model for target domain")
        
        # Balance recommendation
        if row['balance_ratio'] > 4.0:
            print(f"      ⚠️  Imbalanced segments detected")
            print(f"      → Consider: Resampling or trying different k")
    
    # Best practices summary
    print("\n7️⃣ BEST PERFORMERS:")
    print(f"\n   🏆 Best Overall Quality:")
    best_overall = df_summary.loc[
        (df_summary['silhouette_score'] > 0.35) & 
        (df_summary['davies_bouldin_index'] < 1.5)
    ]
    if len(best_overall) > 0:
        for _, row in best_overall.iterrows():
            print(f"      • {row['domain_name']}")
            print(f"        Silhouette: {row['silhouette_score']:.3f}, DB: {row['davies_bouldin_index']:.3f}")
    else:
        print(f"      No domains meet excellent criteria")
        print(f"      Best available: {df_summary.loc[df_summary['silhouette_score'].idxmax(), 'domain_name']}")


def main():
    """
    Main training pipeline for all domains with improved analysis
    """
    print("\n" + "🚀"*40)
    print("RFM CUSTOMER SEGMENTATION - IMPROVED BASELINE CLUSTERING PIPELINE")
    print("🚀"*40)
    print(f"\nRFM Features: {RFM_FEATURES}")
    print(f"K range tested: {K_RANGE}")
    print(f"Domains to train: {len(DOMAINS)}")
    print(f"Algorithm: K-Means with optimal k selection (Silhouette Score)")
    print(f"\n✨ Improvements:")
    print(f"   • Better segment naming (Champions, At Risk, etc.)")
    print(f"   • Value scoring for each segment")
    print(f"   • Enhanced quality interpretation")
    print(f"   • Improved visualizations")
    print(f"   • Transfer learning recommendations")
    
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
        summary_df.to_csv('results/improved_baseline_performance.csv', index=False)
        print(f"\n✅ Saved comprehensive summary: results/improved_baseline_performance.csv")
    
    # Final summary
    print("\n" + "🎉"*40)
    print("TRAINING COMPLETE!")
    print("🎉"*40)
    print(f"\n📦 Deliverables Created:")
    print(f"   ✅ improved_baseline_models.py (Enhanced RFM clustering)")
    print(f"   ✅ {len(all_results)} trained models (.pkl files)")
    print(f"   ✅ improved_baseline_performance.csv (comprehensive metrics)")
    print(f"   ✅ {len(all_results)} customer segment CSVs (with value scores)")
    print(f"   ✅ {len(all_results)} segment profile CSVs (improved naming)")
    print(f"   ✅ {len(all_results) * 4} enhanced visualizations")
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Customers Segmented: {summary_df['n_customers'].sum():,}")
        print(f"   Average Silhouette Score: {summary_df['silhouette_score'].mean():.3f}")
        print(f"   Average Davies-Bouldin: {summary_df['davies_bouldin_index'].mean():.3f}")
        print(f"   Average Segments per Domain: {summary_df['optimal_k'].mean():.1f}")
        print(f"   Domains with Excellent Quality (Sil > 0.4): {len(summary_df[summary_df['silhouette_score'] > 0.4])}")
        print(f"   Domains with Good Quality (Sil > 0.25): {len(summary_df[summary_df['silhouette_score'] > 0.25])}")
    
    print("\n✅ All files saved in:")
    print(f"   📁 models/       - Trained RFM clustering models")
    print(f"   📁 results/      - Performance metrics, segment profiles, customer assignments")
    print(f"   📁 plots/        - Enhanced visualizations (elbow, 3D RFM, profiles, distribution)")
    
    print("\n🎯 Next Steps:")
    print("   1. Review segment profiles with improved naming")
    print("   2. Analyze improved_baseline_performance.csv")
    print("   3. Compare value scores across segments")
    print("   4. Test transfer learning based on recommendations")
    print("   5. Validate transferability predictions")
    print("   6. Run A/B tests with marketing campaigns per segment")


if __name__ == "__main__":
    main()