"""
Calculate Transferability Scores Using RFM Features
Week 2, Day 6-7 | Member 3 Deliverable

This script re-calculates transferability using actual customer behavior (RFM)
instead of product features. This is the PROPER way to measure if a customer
segmentation model trained on one domain will work on another.

Key Difference from Week 1:
- Week 1: Product features (price, rating, discount)
- Week 2: Customer features (Recency, Frequency, Monetary)

Author: Member 3 (Research Lead)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# RESEARCH-BACKED METRICS (Same as Week 1, but applied to RFM data)
# ============================================================================

def calculate_mmd(X_source, X_target, kernel='rbf', gamma=1.0):
    """
    Maximum Mean Discrepancy (MMD)
    Measures distribution difference in kernel space
    
    Lower = more similar = better transferability
    Range: [0, ∞), typically [0, 2]
    
    Reference: Gretton et al. (2012) "A Kernel Two-Sample Test"
    """
    from sklearn.metrics.pairwise import rbf_kernel
    
    n_source = len(X_source)
    n_target = len(X_target)
    
    # Compute kernel matrices
    K_ss = rbf_kernel(X_source, X_source, gamma=gamma)
    K_tt = rbf_kernel(X_target, X_target, gamma=gamma)
    K_st = rbf_kernel(X_source, X_target, gamma=gamma)
    
    # MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    mmd_squared = (K_ss.sum() / (n_source * n_source) - 
                   2 * K_st.sum() / (n_source * n_target) + 
                   K_tt.sum() / (n_target * n_target))
    
    return np.sqrt(max(0, mmd_squared))


def calculate_js_divergence(X_source, X_target, n_bins=50):
    """
    Jensen-Shannon Divergence (averaged across features)
    Symmetric version of KL divergence
    
    Lower = more similar = better transferability
    Range: [0, 1]
    
    Reference: Lin (1991) "Divergence measures based on the Shannon entropy"
    """
    js_scores = []
    
    for col_idx in range(X_source.shape[1]):
        source_col = X_source[:, col_idx]
        target_col = X_target[:, col_idx]
        
        # Create histograms
        bins = np.linspace(
            min(source_col.min(), target_col.min()),
            max(source_col.max(), target_col.max()),
            n_bins
        )
        
        source_hist, _ = np.histogram(source_col, bins=bins, density=True)
        target_hist, _ = np.histogram(target_col, bins=bins, density=True)
        
        # Normalize to probability distributions
        source_hist = source_hist / (source_hist.sum() + 1e-10)
        target_hist = target_hist / (target_hist.sum() + 1e-10)
        
        # Add small epsilon to avoid log(0)
        source_hist += 1e-10
        target_hist += 1e-10
        
        # Calculate JS divergence
        js_scores.append(jensenshannon(source_hist, target_hist))
    
    return np.mean(js_scores)


def calculate_correlation_stability(X_source, X_target):
    """
    Correlation Stability: How similar are feature correlations?
    
    Higher = more similar = better transferability
    Range: [-1, 1], typically [0, 1]
    
    Reference: Ben-David et al. (2010) "A theory of learning from different domains"
    """
    corr_source = np.corrcoef(X_source.T)
    corr_target = np.corrcoef(X_target.T)
    
    # Flatten correlation matrices (excluding diagonal)
    n = corr_source.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    source_flat = corr_source[mask]
    target_flat = corr_target[mask]
    
    # Pearson correlation between correlation structures
    stability = np.corrcoef(source_flat, target_flat)[0, 1]
    
    return stability


def calculate_ks_statistic(X_source, X_target):
    """
    Kolmogorov-Smirnov Statistic (averaged across features)
    Tests if two samples come from same distribution
    
    Lower = more similar = better transferability
    Range: [0, 1]
    
    Reference: Massey (1951) "The Kolmogorov-Smirnov Test"
    """
    ks_scores = []
    
    for col_idx in range(X_source.shape[1]):
        statistic, _ = ks_2samp(X_source[:, col_idx], X_target[:, col_idx])
        ks_scores.append(statistic)
    
    return np.mean(ks_scores)


def calculate_wasserstein(X_source, X_target):
    """
    Wasserstein Distance (averaged across features)
    Earth Mover's Distance - optimal transport cost
    
    Lower = more similar = better transferability
    Range: [0, ∞), scale depends on feature values
    
    Reference: Villani (2009) "Optimal Transport: Old and New"
    """
    wasserstein_scores = []
    
    for col_idx in range(X_source.shape[1]):
        distance = wasserstein_distance(
            X_source[:, col_idx], 
            X_target[:, col_idx]
        )
        wasserstein_scores.append(distance)
    
    return np.mean(wasserstein_scores)


def calculate_composite_transferability_score(
    mmd, js_div, corr_stability, ks_stat, wasserstein,
    weights=None
):
    """
    Combine multiple metrics into single transferability score
    
    Score range: [0, 1] where higher = better transferability
    
    Default weights based on literature review:
    - MMD: 35% (strongest predictor in domain adaptation)
    - JS Divergence: 25% (robust distribution comparison)
    - Correlation Stability: 20% (captures feature relationships)
    - KS Statistic: 10% (simple but effective)
    - Wasserstein: 10% (optimal transport perspective)
    """
    if weights is None:
        weights = {
            'mmd': 0.35,
            'js_div': 0.25,
            'corr_stability': 0.20,
            'ks_stat': 0.10,
            'wasserstein': 0.10
        }
    
    # Normalize metrics to [0, 1] where higher = better
    # For distance metrics (MMD, JS, KS, Wasserstein): invert them
    mmd_norm = 1 / (1 + mmd)  # Closer to 1 when MMD is small
    js_norm = 1 - js_div  # JS is already in [0,1]
    ks_norm = 1 - ks_stat  # KS is in [0,1]
    
    # Wasserstein needs scaling (depends on feature scale)
    # Use sigmoid to map to [0,1]
    wasserstein_norm = 1 / (1 + wasserstein)
    
    # Correlation stability is already in [-1,1], map to [0,1]
    corr_norm = (corr_stability + 1) / 2
    
    # Weighted combination
    score = (weights['mmd'] * mmd_norm +
             weights['js_div'] * js_norm +
             weights['corr_stability'] * corr_norm +
             weights['ks_stat'] * ks_norm +
             weights['wasserstein'] * wasserstein_norm)
    
    return score


# ============================================================================
# MAIN CALCULATION FUNCTION
# ============================================================================

def calculate_transferability_for_pair(source_df, target_df, pair_name):
    """
    Calculate all transferability metrics for a domain pair
    
    Args:
        source_df: DataFrame with RFM features for source domain
        target_df: DataFrame with RFM features for target domain
        pair_name: String identifier (e.g., "Pair 1: Cleaning → Foodgrains")
    
    Returns:
        Dictionary with all metrics
    """
    print(f"\n{'='*70}")
    print(f"Calculating Transferability: {pair_name}")
    print(f"{'='*70}")
    
    # Extract RFM features
    rfm_features = ['Recency', 'Frequency', 'Monetary']
    
    X_source = source_df[rfm_features].values
    X_target = target_df[rfm_features].values
    
    print(f"Source customers: {len(X_source):,}")
    print(f"Target customers: {len(X_target):,}")
    
    # Standardize features (important for distance metrics)
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    # Calculate all metrics
    print("Calculating metrics...")
    
    mmd = calculate_mmd(X_source_scaled, X_target_scaled)
    js_div = calculate_js_divergence(X_source_scaled, X_target_scaled)
    corr_stability = calculate_correlation_stability(X_source_scaled, X_target_scaled)
    ks_stat = calculate_ks_statistic(X_source_scaled, X_target_scaled)
    wasserstein = calculate_wasserstein(X_source_scaled, X_target_scaled)
    
    # Composite score
    transfer_score = calculate_composite_transferability_score(
        mmd, js_div, corr_stability, ks_stat, wasserstein
    )
    
    # Print results
    print(f"\n📊 Transferability Metrics:")
    print(f"  MMD:                 {mmd:.4f} (lower = better)")
    print(f"  JS Divergence:       {js_div:.4f} (lower = better)")
    print(f"  Correlation Stability: {corr_stability:.4f} (higher = better)")
    print(f"  KS Statistic:        {ks_stat:.4f} (lower = better)")
    print(f"  Wasserstein:         {wasserstein:.4f} (lower = better)")
    print(f"\n🎯 Composite Transferability Score: {transfer_score:.4f}")
    
    # Interpretation
    if transfer_score >= 0.75:
        recommendation = "HIGH - Transfer as-is"
        strategy = "Use source model directly with minimal/no fine-tuning"
    elif transfer_score >= 0.55:
        recommendation = "MODERATE - Fine-tune recommended"
        strategy = "Use 10-30% of target data for fine-tuning"
    else:
        recommendation = "LOW - Train new model"
        strategy = "Train separate model on target domain"
    
    print(f"\n✅ Recommendation: {recommendation}")
    print(f"   Strategy: {strategy}")
    
    return {
        'pair_name': pair_name,
        'n_source_customers': len(X_source),
        'n_target_customers': len(X_target),
        'mmd': mmd,
        'js_divergence': js_div,
        'correlation_stability': corr_stability,
        'ks_statistic': ks_stat,
        'wasserstein_distance': wasserstein,
        'transferability_score': transfer_score,
        'recommendation': recommendation,
        'strategy': strategy
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("WEEK 2: TRANSFERABILITY ANALYSIS WITH RFM FEATURES")
    print("Research-Backed Metrics Applied to Customer Behavior Data")
    print("="*80)
    
    # Define paths
    data_dir = Path('.')
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Domain pair definitions
    pairs = [
        {
            'name': 'Pair 1: Cleaning & Household → Foodgrains, Oil & Masala',
            'source_file': 'domain_pair1_source_RFM.csv',
            'target_file': 'domain_pair1_target_RFM.csv',
            'expected': 'HIGH transferability (similar customer behavior)'
        },
        {
            'name': 'Pair 2: Snacks & Branded Foods → Fruits & Vegetables',
            'source_file': 'domain_pair2_source_RFM.csv',
            'target_file': 'domain_pair2_target_RFM.csv',
            'expected': 'MODERATE transferability (different purchase patterns)'
        },
        {
            'name': 'Pair 3: Premium Segment → Budget Segment',
            'source_file': 'domain_pair3_source_RFM.csv',
            'target_file': 'domain_pair3_target_RFM.csv',
            'expected': 'LOW transferability (different customer segments)'
        },
        {
            'name': 'Pair 4: Popular Brands → Niche Brands',
            'source_file': 'domain_pair4_source_RFM.csv',
            'target_file': 'domain_pair4_target_RFM.csv',
            'expected': 'LOW-MODERATE transferability (brand loyalty differences)'
        }
    ]
    
    # Calculate transferability for each pair
    results = []
    
    for pair in pairs:
        print(f"\n\n{'#'*80}")
        print(f"Processing: {pair['name']}")
        print(f"Expected: {pair['expected']}")
        print(f"{'#'*80}")
        
        # Load RFM data
        source_path = data_dir / pair['source_file']
        target_path = data_dir / pair['target_file']
        
        if not source_path.exists() or not target_path.exists():
            print(f"⚠️  WARNING: Files not found!")
            print(f"   Expected: {source_path}")
            print(f"   Expected: {target_path}")
            continue
        
        source_df = pd.read_csv(source_path)
        target_df = pd.read_csv(target_path)
        
        # Calculate metrics
        result = calculate_transferability_for_pair(
            source_df, target_df, pair['name']
        )
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file = output_dir / 'transferability_scores_with_RFM.csv'
    results_df.to_csv(output_file, index=False)
    
    print("\n\n" + "="*80)
    print("📊 SUMMARY: TRANSFERABILITY SCORES (RFM-Based)")
    print("="*80)
    print(results_df[['pair_name', 'transferability_score', 'recommendation']].to_string(index=False))
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Compare with Week 1 predictions
    print("\n" + "="*80)
    print("🔄 COMPARISON: Week 1 (Product Features) vs Week 2 (Customer Behavior)")
    print("="*80)
    print("\nExpected Insight:")
    print("  Week 1 analyzed product similarity (price, ratings)")
    print("  Week 2 analyzes customer behavior similarity (RFM)")
    print("  Scores may differ because customer segments can behave differently")
    print("  even when buying similar products!")
    print("\nExample:")
    print("  Cleaning products & Food grains may have similar prices (Week 1: High)")
    print("  But customers who buy cleaning products may shop less frequently")
    print("  than customers who buy food grains (Week 2: Lower?)")
    print("  → This affects whether a segmentation model transfers well!")
    
    print("\n✅ Week 2 Day 6-7 Complete!")
    print("📁 Next: Use these scores to validate experiments (Week 3)")


if __name__ == "__main__":
    main()