# WEEK 1 FIXED - TRANSFERABILITY ANALYSIS WITH UPDATED DATASET
# Uses BigBasket_v3.csv (with processed Eggs/Meat/Fish, no Fruits & Vegetables)
# No preprocessing - data already cleaned
# Author: Transfer Learning Framework Team
# Date: November 6, 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("="*80)
print("WEEK 1 FIXED - TRANSFERABILITY ANALYSIS")
print("Using BigBasket_v3.csv (Processed Eggs/Meat/Fish, No Fruits & Vegetables)")
print("="*80)

# ============================================================================
# LOAD PREPROCESSED DATA (NO CLEANING NEEDED)
# ============================================================================

print("\n📂 Loading BigBasket_v3.csv...")
df = pd.read_csv('data/processed/BigBasket_v3.csv')
print(f"✓ Loaded {len(df):,} products")
print(f"✓ Columns: {list(df.columns)}")

# ============================================================================
# CATEGORY OVERVIEW
# ============================================================================

print("\n" + "🟢"*30)
print("CATEGORY OVERVIEW")
print("🟢"*30)

all_categories = df['category'].unique()
print(f"\n📦 Total categories: {len(all_categories)}")

# Category statistics
category_stats = []
for cat in all_categories:
    cat_df = df[df['category'] == cat]
    stats = {
        'category': cat,
        'n_products': len(cat_df),
        'pct': len(cat_df) / len(df) * 100,
        'avg_price': cat_df['sale_price'].mean(),
        'median_price': cat_df['sale_price'].median(),
        'avg_rating': cat_df['rating_clean'].mean(),
        'n_brands': cat_df['brand'].nunique() if 'brand' in cat_df.columns else 0
    }
    category_stats.append(stats)

stats_df = pd.DataFrame(category_stats).sort_values('n_products', ascending=False)

print("\n" + "="*80)
print(f"{'Category':<40} {'Products':>8} {'%':>6} {'Avg Price':>10} {'Rating':>7}")
print("-"*80)
for _, row in stats_df.iterrows():
    print(f"{row['category']:<40} {row['n_products']:>8,} {row['pct']:>5.1f}% ₹{row['avg_price']:>8.2f} {row['avg_rating']:>7.2f}")

stats_df.to_csv('src/week1/category_statistics_FIXED.csv', index=False)
print(f"\n✓ Saved: src/week1/category_statistics_FIXED.csv")

# ============================================================================
# TRANSFERABILITY METRIC FUNCTIONS
# ============================================================================

def compute_mmd(X_source, X_target, gamma=1.0):
    """Maximum Mean Discrepancy - Gold standard in domain adaptation"""
    def rbf_kernel(X, Y, gamma):
        XX = np.dot(X, X.T)
        YY = np.dot(Y, Y.T)
        XY = np.dot(X, Y.T)
        X_sqnorms = np.diag(XX)
        Y_sqnorms = np.diag(YY)
        K = np.exp(-gamma * (X_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * XY))
        return K
    
    n_source = X_source.shape[0]
    n_target = X_target.shape[0]
    
    K_ss = rbf_kernel(X_source, X_source, gamma)
    K_tt = rbf_kernel(X_target, X_target, gamma)
    K_st = rbf_kernel(X_source, X_target, gamma)
    
    mmd_squared = (
        np.sum(K_ss) / (n_source ** 2) +
        np.sum(K_tt) / (n_target ** 2) -
        2 * np.sum(K_st) / (n_source * n_target)
    )
    
    return np.sqrt(max(mmd_squared, 0))

def compute_js_divergence(X_source, X_target, n_bins=50):
    """Jensen-Shannon Divergence (symmetric KL)"""
    n_features = X_source.shape[1]
    js_scores = []
    
    for i in range(n_features):
        min_val = min(X_source[:, i].min(), X_target[:, i].min())
        max_val = max(X_source[:, i].max(), X_target[:, i].max())
        
        # Handle constant features
        if min_val == max_val:
            js_scores.append(0.0)
            continue
            
        bins = np.linspace(min_val, max_val, n_bins)
        
        hist_s, _ = np.histogram(X_source[:, i], bins=bins, density=True)
        hist_t, _ = np.histogram(X_target[:, i], bins=bins, density=True)
        
        hist_s = hist_s / (hist_s.sum() + 1e-10)
        hist_t = hist_t / (hist_t.sum() + 1e-10)
        
        js = jensenshannon(hist_s, hist_t)
        # Handle NaN values
        if np.isnan(js):
            js = 0.0
        js_scores.append(js)
    
    return np.mean(js_scores)

def compute_correlation_stability(X_source, X_target):
    """Measure correlation matrix similarity"""
    if X_source.shape[1] < 2:
        return 1.0  # Perfect stability if only 1 feature
    
    try:
        corr_s = np.corrcoef(X_source.T)
        corr_t = np.corrcoef(X_target.T)
        
        # Handle NaN in correlation matrices
        if np.any(np.isnan(corr_s)) or np.any(np.isnan(corr_t)):
            return 0.5  # Return neutral score if correlation can't be computed
        
        diff = corr_s - corr_t
        frobenius = np.sqrt(np.sum(diff ** 2))
        
        n_features = X_source.shape[1]
        max_dist = np.sqrt(2 * n_features ** 2)
        
        stability = 1 - (frobenius / max_dist)
        
        # Ensure result is valid
        if np.isnan(stability) or np.isinf(stability):
            return 0.5
            
        return stability
    except Exception as e:
        print(f"    Warning: Correlation stability computation failed: {e}")
        return 0.5

def compute_ks_statistic(X_source, X_target):
    """Kolmogorov-Smirnov test statistic"""
    n_features = X_source.shape[1]
    ks_scores = []
    
    for i in range(n_features):
        ks_stat, _ = ks_2samp(X_source[:, i], X_target[:, i])
        ks_scores.append(ks_stat)
    
    return np.mean(ks_scores)

def compute_wasserstein(X_source, X_target):
    """Wasserstein distance (Earth Mover's Distance)"""
    n_features = X_source.shape[1]
    w_scores = []
    
    for i in range(n_features):
        w_dist = wasserstein_distance(X_source[:, i], X_target[:, i])
        w_scores.append(w_dist)
    
    return np.mean(w_scores)

def calculate_transferability_score(source_df, target_df, feature_cols):
    """
    Calculate comprehensive transferability score using research-backed metrics
    
    Returns:
        score (0-1): Higher = better transferability
        metrics (dict): Individual metric values
    """
    # Extract features
    X_source = source_df[feature_cols].values
    X_target = target_df[feature_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_s_scaled = scaler.fit_transform(X_source)
    X_t_scaled = scaler.transform(X_target)
    
    # Calculate metrics with error handling
    try:
        mmd = compute_mmd(X_s_scaled, X_t_scaled)
    except Exception as e:
        print(f"    Warning: MMD computation failed: {e}")
        mmd = 1.0
    
    try:
        js = compute_js_divergence(X_s_scaled, X_t_scaled)
    except Exception as e:
        print(f"    Warning: JS divergence computation failed: {e}")
        js = 1.0
    
    try:
        corr = compute_correlation_stability(X_s_scaled, X_t_scaled)
    except Exception as e:
        print(f"    Warning: Correlation stability computation failed: {e}")
        corr = 0.5
    
    try:
        ks = compute_ks_statistic(X_s_scaled, X_t_scaled)
    except Exception as e:
        print(f"    Warning: KS statistic computation failed: {e}")
        ks = 1.0
    
    try:
        w_dist = compute_wasserstein(X_s_scaled, X_t_scaled)
    except Exception as e:
        print(f"    Warning: Wasserstein computation failed: {e}")
        w_dist = 1.0
    
    # Normalize to 0-1 (higher = better)
    mmd_norm = max(0, 1 - mmd / 2.0)
    js_norm = 1 - js
    corr_norm = corr
    ks_norm = 1 - ks
    w_norm = max(0, 1 - w_dist / 2.0)
    
    # Weighted combination (based on literature)
    score = (
        0.35 * mmd_norm +
        0.25 * js_norm +
        0.20 * corr_norm +
        0.10 * ks_norm +
        0.10 * w_norm
    )
    
    metrics = {
        'mmd': mmd,
        'js_divergence': js,
        'correlation_stability': corr,
        'ks_statistic': ks,
        'wasserstein': w_dist
    }
    
    return score, metrics

# ============================================================================
# CALCULATE TRANSFERABILITY FOR ALL CATEGORY PAIRS
# ============================================================================

print("\n" + "🟡"*30)
print("TRANSFERABILITY ANALYSIS")
print("🟡"*30)

# Define features for analysis
feature_cols = ['sale_price', 'rating_clean']
if 'discount_pct' in df.columns:
    feature_cols.append('discount_pct')

print(f"\n📊 Features used: {feature_cols}")
print(f"📊 Analyzing all {len(list(combinations(all_categories, 2)))} category pairs...")

transferability_results = []
failed_pairs = []

for cat1, cat2 in combinations(all_categories, 2):
    source_df = df[df['category'] == cat1]
    target_df = df[df['category'] == cat2]
    
    # Skip if too few samples
    if len(source_df) < 30 or len(target_df) < 30:
        failed_pairs.append({
            'source': cat1,
            'target': cat2,
            'reason': f'Insufficient samples (source={len(source_df)}, target={len(target_df)})'
        })
        continue
    
    try:
        score, metrics = calculate_transferability_score(
            source_df, target_df, feature_cols
        )
        
        # Validate that all metrics are computed successfully
        if any(v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) 
               for v in [score] + list(metrics.values())):
            failed_pairs.append({
                'source': cat1,
                'target': cat2,
                'reason': 'Invalid metrics (NaN or Inf values)'
            })
            print(f"  ⚠️  Invalid metrics for {cat1} → {cat2}, skipping")
            continue
        
        result = {
            'source': cat1,
            'target': cat2,
            'transferability_score': score,
            'mmd': metrics.get('mmd'),
            'js_divergence': metrics.get('js_divergence'),
            'correlation_stability': metrics.get('correlation_stability'),
            'ks_statistic': metrics.get('ks_statistic'),
            'wasserstein': metrics.get('wasserstein'),
            'n_source': len(source_df),
            'n_target': len(target_df),
            'avg_price_source': source_df['sale_price'].mean(),
            'avg_price_target': target_df['sale_price'].mean()
        }
        transferability_results.append(result)
        
    except Exception as e:
        failed_pairs.append({
            'source': cat1,
            'target': cat2,
            'reason': f'Error: {str(e)}'
        })
        print(f"  ⚠️  Error computing metrics for {cat1} → {cat2}: {e}")
        continue

# Convert to DataFrame
transfer_df = pd.DataFrame(transferability_results)
transfer_df = transfer_df.sort_values('transferability_score', ascending=False)

# Save results
transfer_df.to_csv('src/week1/transferability_scores_FIXED.csv', index=False)
print(f"\n✓ Calculated transferability for {len(transfer_df)} category pairs")
print(f"✓ Saved: src/week1/transferability_scores_FIXED.csv")

# Save failed pairs if any
if failed_pairs:
    failed_df = pd.DataFrame(failed_pairs)
    failed_df.to_csv('src/week1/failed_pairs_FIXED.csv', index=False)
    print(f"⚠️  {len(failed_pairs)} pairs failed - saved to: src/week1/failed_pairs_FIXED.csv")

# ============================================================================
# CREATE TRANSFERABILITY HEATMAP
# ============================================================================

print("\n" + "🟠"*30)
print("CREATING TRANSFERABILITY HEATMAP")
print("🟠"*30)

fig, ax = plt.subplots(figsize=(14, 12))

# Create similarity matrix
n_cats = len(all_categories)
similarity_matrix = np.zeros((n_cats, n_cats))
cat_to_idx = {cat: i for i, cat in enumerate(sorted(all_categories))}

for _, row in transfer_df.iterrows():
    i = cat_to_idx[row['source']]
    j = cat_to_idx[row['target']]
    score = row['transferability_score']
    similarity_matrix[i, j] = score
    similarity_matrix[j, i] = score

np.fill_diagonal(similarity_matrix, 1.0)

# Create heatmap
sorted_cats = sorted(all_categories)
sns.heatmap(
    similarity_matrix,
    xticklabels=sorted_cats,
    yticklabels=sorted_cats,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Transferability Score'},
    square=True,
    ax=ax,
    annot_kws={'size': 8}
)

ax.set_title('Category Transferability Matrix - FIXED\n(With Processed Eggs/Meat/Fish, No Fruits & Vegetables)\nGreen = High Transfer | Red = Low Transfer', 
             fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('src/week1/transferability_heatmap_FIXED.png', dpi=300, bbox_inches='tight')
print("✓ Saved: src/week1/transferability_heatmap_FIXED.png")
plt.close()

# Create top 15 pairs bar chart
fig, ax = plt.subplots(figsize=(14, 8))
top15 = transfer_df.head(15).copy()
top15['pair'] = top15['source'].str[:20] + ' → ' + top15['target'].str[:20]

colors = ['green' if s > 0.7 else 'orange' if s > 0.5 else 'red' 
          for s in top15['transferability_score']]

ax.barh(range(len(top15)), top15['transferability_score'], color=colors, alpha=0.7)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['pair'], fontsize=10)
ax.set_xlabel('Transferability Score', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Category Pairs by Transferability - FIXED', 
             fontsize=14, fontweight='bold')
ax.axvline(0.7, color='green', linestyle='--', alpha=0.3, label='High (>0.7)')
ax.axvline(0.5, color='orange', linestyle='--', alpha=0.3, label='Moderate (>0.5)')
ax.legend()
plt.tight_layout()
plt.savefig('src/week1/top15_transferability_pairs_FIXED.png', dpi=300, bbox_inches='tight')
print("✓ Saved: src/week1/top15_transferability_pairs_FIXED.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 WEEK 1 FIXED - ANALYSIS COMPLETE!")
print("="*80)

print("\n📦 SUMMARY:")
print(f"  Total products: {len(df):,}")
print(f"  Categories analyzed: {len(all_categories)}")
print(f"  Category pairs evaluated: {len(transfer_df)}")
print(f"  Failed pairs: {len(failed_pairs)}")
print(f"  Features used: {feature_cols}")

print("\n✅ FILES CREATED:")
print("  1. src/week1/category_statistics_FIXED.csv")
print("  2. src/week1/transferability_scores_FIXED.csv")
print("  3. src/week1/transferability_heatmap_FIXED.png")
print("  4. src/week1/top15_transferability_pairs_FIXED.png")
if failed_pairs:
    print("  5. src/week1/failed_pairs_FIXED.csv")

print("\n🏆 TOP 5 TRANSFERABLE PAIRS:")
for idx, row in transfer_df.head(5).iterrows():
    print(f"  {row['source']:<35} → {row['target']:<35} Score: {row['transferability_score']:.3f}")

print("\n📚 METRICS USED:")
print("  • MMD (Maximum Mean Discrepancy) - 35% weight")
print("  • JS Divergence (Jensen-Shannon) - 25% weight")
print("  • Correlation Stability - 20% weight")
print("  • KS Statistic (Kolmogorov-Smirnov) - 10% weight")
print("  • Wasserstein Distance - 10% weight")

print("\n" + "="*80)
print("✅ Ready for Week 2!")
print("="*80)
