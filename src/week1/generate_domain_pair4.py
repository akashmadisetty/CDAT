"""
Generate Domain Pair 4: Premium → Mass-Market Beauty Brands
FIXED VERSION: Added brand_premium_index and price_retention features
Removed rating_clean (not discriminative)

Author: Transfer Learning Framework Team
Date: November 6, 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DOMAIN PAIR 4 GENERATION (FIXED VERSION)")
print("Premium → Mass-Market Beauty Brands")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📂 Loading BigBasket_v3.csv...")
df = pd.read_csv('data/processed/BigBasket_v3.csv')
print(f"✓ Loaded {len(df):,} products")

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
        
        if min_val == max_val:
            js_scores.append(0.0)
            continue
            
        bins = np.linspace(min_val, max_val, n_bins)
        
        hist_s, _ = np.histogram(X_source[:, i], bins=bins, density=True)
        hist_t, _ = np.histogram(X_target[:, i], bins=bins, density=True)
        
        hist_s = hist_s / (hist_s.sum() + 1e-10)
        hist_t = hist_t / (hist_t.sum() + 1e-10)
        
        js = jensenshannon(hist_s, hist_t)
        if np.isnan(js):
            js = 0.0
        js_scores.append(js)
    
    return np.mean(js_scores)

def compute_correlation_stability(X_source, X_target):
    """Measure correlation matrix similarity"""
    if X_source.shape[1] < 2:
        return 1.0
    
    try:
        corr_s = np.corrcoef(X_source.T)
        corr_t = np.corrcoef(X_target.T)
        
        if np.any(np.isnan(corr_s)) or np.any(np.isnan(corr_t)):
            return 0.5
        
        diff = corr_s - corr_t
        frobenius = np.sqrt(np.sum(diff ** 2))
        
        n_features = X_source.shape[1]
        max_dist = np.sqrt(2 * n_features ** 2)
        
        stability = 1 - (frobenius / max_dist)
        
        if np.isnan(stability) or np.isinf(stability):
            return 0.5
            
        return stability
    except Exception:
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
    """Calculate comprehensive transferability score"""
    X_source = source_df[feature_cols].values
    X_target = target_df[feature_cols].values
    
    scaler = StandardScaler()
    X_s_scaled = scaler.fit_transform(X_source)
    X_t_scaled = scaler.transform(X_target)
    
    mmd = compute_mmd(X_s_scaled, X_t_scaled)
    js = compute_js_divergence(X_s_scaled, X_t_scaled)
    corr = compute_correlation_stability(X_s_scaled, X_t_scaled)
    ks = compute_ks_statistic(X_s_scaled, X_t_scaled)
    w_dist = compute_wasserstein(X_s_scaled, X_t_scaled)
    
    # Normalize to 0-1 (higher = better)
    mmd_norm = max(0, 1 - mmd / 2.0)
    js_norm = 1 - js
    corr_norm = corr
    ks_norm = 1 - ks
    w_norm = max(0, 1 - w_dist / 2.0)
    
    # Weighted combination
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
# PAIR 4: PREMIUM → MASS-MARKET BRANDS (Within Beauty & Hygiene)
# ============================================================================

print("\n" + "="*60)
print("❌ DOMAIN PAIR 4: LOW - PREMIUM → MASS-MARKET BEAUTY BRANDS")
print("="*60)

# Step 1: Focus on Beauty & Hygiene only
print("\n📦 Step 1: Filtering Beauty & Hygiene category...")
beauty_df = df[df['category'] == 'Beauty & Hygiene'].copy()
print(f"✓ Found {len(beauty_df):,} Beauty & Hygiene products")

if len(beauty_df) == 0:
    print("\n❌ ERROR: Beauty & Hygiene category not found!")
    exit(1)

# Step 2: Define premium vs mass-market by PRICE (AGGRESSIVE)
print("\n💰 Step 2: Segmenting by price...")
print("   Premium: Top 10% price range (≥ 90th percentile)")
print("   Mass-market: Bottom 25% price range (≤ 25th percentile)")

price_90 = beauty_df['sale_price'].quantile(0.90)  # ULTRA-PREMIUM
price_25 = beauty_df['sale_price'].quantile(0.25)  # BUDGET

premium_beauty = beauty_df[beauty_df['sale_price'] >= price_90].copy()
mass_market_beauty = beauty_df[beauty_df['sale_price'] <= price_25].copy()

print(f"\nPremium (≥90th): ₹{price_90:.2f}")
print(f"Mass-market (≤25th): ₹{price_25:.2f}")

# Step 3: FEATURE ENGINEERING
print("\n🔧 Step 3: Engineering discriminative features...")

# 3a. Brand Premium Index (captures brand positioning)
brand_avg_prices = beauty_df.groupby('brand')['sale_price'].mean()
premium_beauty['brand_premium_index'] = premium_beauty['brand'].map(brand_avg_prices)
mass_market_beauty['brand_premium_index'] = mass_market_beauty['brand'].map(brand_avg_prices)

# 3b. Price Retention (sale_price / market_price)
# Premium brands discount less, mass-market discounts aggressively
premium_beauty['price_retention'] = premium_beauty['sale_price'] / premium_beauty['market_price']
mass_market_beauty['price_retention'] = mass_market_beauty['sale_price'] / mass_market_beauty['market_price']

print("✓ Created features:")
print("   • brand_premium_index (brand positioning)")
print("   • price_retention (pricing strategy)")

# Step 4: Get brand composition
print("\n🏷️  Step 4: Analyzing domain characteristics...")
premium_brands = premium_beauty['brand'].value_counts()
mass_brands = mass_market_beauty['brand'].value_counts()
brand_overlap = len(set(premium_brands.index) & set(mass_brands.index))

print(f"\nCategory: Beauty & Hygiene")
print(f"\n{'='*60}")
print(f"SOURCE DOMAIN: Premium Beauty Brands")
print(f"{'='*60}")
print(f"  Products: {len(premium_beauty):,}")
print(f"  Avg Price: ₹{premium_beauty['sale_price'].mean():.2f}")
print(f"  Price Range: ₹{premium_beauty['sale_price'].min():.0f} - ₹{premium_beauty['sale_price'].max():.0f}")
print(f"  Unique Brands: {premium_beauty['brand'].nunique()}")
print(f"  Avg Brand Index: ₹{premium_beauty['brand_premium_index'].mean():.2f}")
print(f"  Avg Price Retention: {premium_beauty['price_retention'].mean():.3f}")

print(f"\n{'='*60}")
print(f"TARGET DOMAIN: Mass-Market Beauty Brands")
print(f"{'='*60}")
print(f"  Products: {len(mass_market_beauty):,}")
print(f"  Avg Price: ₹{mass_market_beauty['sale_price'].mean():.2f}")
print(f"  Price Range: ₹{mass_market_beauty['sale_price'].min():.0f} - ₹{mass_market_beauty['sale_price'].max():.0f}")
print(f"  Unique Brands: {mass_market_beauty['brand'].nunique()}")
print(f"  Avg Brand Index: ₹{mass_market_beauty['brand_premium_index'].mean():.2f}")
print(f"  Avg Price Retention: {mass_market_beauty['price_retention'].mean():.3f}")

print(f"\n{'='*60}")
print(f"Brand Overlap: {brand_overlap} brands ({brand_overlap/max(len(premium_brands),1)*100:.1f}%)")

# Step 5: Calculate transferability WITH NEW FEATURES
print("\n📊 Step 5: Calculating transferability score...")

# UPDATED FEATURES (NO RATING!)
feature_cols = [
    'sale_price',           # Base price difference
    'discount_pct',         # Discount aggressiveness  
    'brand_premium_index',  # Brand positioning strategy
    'price_retention'       # Pricing strategy (discount vs no-discount)
]

print(f"   Features used: {feature_cols}")
print(f"   ⚠️  NOTE: Removed 'rating_clean' (not discriminative)")

score4, metrics4 = calculate_transferability_score(
    premium_beauty, 
    mass_market_beauty, 
    feature_cols
)

print(f"\n{'='*60}")
print(f"TRANSFERABILITY RESULTS")
print(f"{'='*60}")
print(f"\n📈 Overall Score: {score4:.3f}")
print(f"\n📊 Individual Metrics:")
print(f"   MMD: {metrics4['mmd']:.4f} (lower = better)")
print(f"   JS Divergence: {metrics4['js_divergence']:.4f} (lower = better)")
print(f"   Correlation Stability: {metrics4['correlation_stability']:.3f} (higher = better)")
print(f"   KS Statistic: {metrics4['ks_statistic']:.4f} (lower = better)")
print(f"   Wasserstein: {metrics4['wasserstein']:.4f} (lower = better)")

print(f"\n🎯 Target Score: LOW-MODERATE (0.35-0.50)")
print(f"   Reason: Premium customers seek luxury/status")
print(f"           Mass-market customers seek value/function")
print(f"           Purchase behaviors differ fundamentally")

# Step 6: Export domain pairs
print(f"\n💾 Step 6: Exporting domain pair files...")
output_dir = 'data/domains'
import os
os.makedirs(output_dir, exist_ok=True)

source_file = os.path.join(output_dir, 'domain_pair4_source.csv')
target_file = os.path.join(output_dir, 'domain_pair4_target.csv')

premium_beauty.to_csv(source_file, index=False)
mass_market_beauty.to_csv(target_file, index=False)

print(f"✓ Exported: {source_file}")
print(f"✓ Exported: {target_file}")

# Diagnostic info
print("\n" + "="*60)
print("🔍 DIAGNOSTIC CHECK")
print("="*60)
print(f"Premium avg price: ₹{premium_beauty['sale_price'].mean():.2f}")
print(f"Mass-market avg price: ₹{mass_market_beauty['sale_price'].mean():.2f}")
print(f"Price ratio: {premium_beauty['sale_price'].mean() / mass_market_beauty['sale_price'].mean():.2f}x")

print(f"\nFeature Differences:")
print(f"  Brand Index Ratio: {premium_beauty['brand_premium_index'].mean() / mass_market_beauty['brand_premium_index'].mean():.2f}x")
print(f"  Price Retention Diff: {abs(premium_beauty['price_retention'].mean() - mass_market_beauty['price_retention'].mean()):.3f}")

print(f"\n💡 Interpretation:")
if score4 > 0.60:
    print(f"   ⚠️  Score > 0.60: Still moderate similarity")
    print(f"   → Consider cross-category pair (Baby Care → Beverages)")
elif 0.40 <= score4 <= 0.55:
    print(f"   ✅ Score 0.40-0.55: LOW-MODERATE (Perfect!)")
    print(f"   → Framework should recommend heavy fine-tuning or train new")
elif score4 < 0.40:
    print(f"   ✅ Score < 0.40: LOW transferability (Excellent!)")
    print(f"   → Framework should recommend training new model")
else:
    print(f"   ℹ️  Score 0.55-0.60: Moderate transferability")
    print(f"   → Some transfer possible with fine-tuning")

# Final summary
print("\n" + "="*80)
print("✅ DOMAIN PAIR 4 GENERATION COMPLETE!")
print("="*80)

print(f"\n📦 Summary:")
print(f"   Source (Premium): {len(premium_beauty):,} products")
print(f"   Target (Mass-market): {len(mass_market_beauty):,} products")
print(f"   Transferability Score: {score4:.3f}")
if score4 < 0.50:
    interpretation = "LOW"
elif score4 < 0.60:
    interpretation = "LOW-MODERATE"
else:
    interpretation = "MODERATE"
print(f"   Interpretation: {interpretation}")

print(f"\n📁 Files created:")
print(f"   • {source_file}")
print(f"   • {target_file}")

if score4 < 0.50:
    print("\n🎯 Recommendation: TRAIN NEW MODEL")
    print("   Transfer learning not recommended - domains too different")
else:
    print("\n🎯 Recommendation: HEAVY FINE-TUNING (40-50% target data)")
    print("   Limited transfer benefit - fine-tuning required")