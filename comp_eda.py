# Complete BigBasket EDA - Analyze ALL Categories
# Auto-select best domain pairs for transfer learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("COMPLETE BIGBASKET EDA - ALL CATEGORIES ANALYSIS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv('BigBasket_Products.csv')
print(f"\n✓ Loaded {len(df):,} products")

# ============================================================================
# STEP 1: ANALYZE ALL CATEGORIES
# ============================================================================

print("\n" + "="*80)
print("STEP 1: COMPREHENSIVE CATEGORY ANALYSIS")
print("="*80)

all_categories = df['category'].unique()
print(f"\n📦 Found {len(all_categories)} categories")

# Store detailed stats for each category
category_stats = []

for cat in all_categories:
    cat_df = df[df['category'] == cat].copy()
    
    # Clean rating
    cat_df['rating_clean'] = pd.to_numeric(cat_df['rating'], errors='coerce')
    
    stats = {
        'category': cat,
        'n_products': len(cat_df),
        'pct_of_total': len(cat_df) / len(df) * 100,
        'avg_price': cat_df['sale_price'].mean(),
        'median_price': cat_df['sale_price'].median(),
        'std_price': cat_df['sale_price'].std(),
        'min_price': cat_df['sale_price'].min(),
        'max_price': cat_df['sale_price'].max(),
        'n_brands': cat_df['brand'].nunique(),
        'avg_rating': cat_df['rating_clean'].mean(),
        'n_subcategories': cat_df['sub_category'].nunique() if 'sub_category' in cat_df.columns else 0
    }
    category_stats.append(stats)

# Convert to DataFrame
stats_df = pd.DataFrame(category_stats)
stats_df = stats_df.sort_values('n_products', ascending=False)

print("\n📊 ALL CATEGORIES - DETAILED STATISTICS:")
print("="*80)
print(f"{'Category':<35} {'Products':>8} {'%':>6} {'Avg Price':>10} {'Brands':>7} {'Rating':>7}")
print("-"*80)
for _, row in stats_df.iterrows():
    print(f"{row['category']:<35} {row['n_products']:>8,} {row['pct_of_total']:>5.1f}% ₹{row['avg_price']:>8.2f} {row['n_brands']:>7} {row['avg_rating']:>7.2f}")

# Save category statistics
stats_df.to_csv('category_statistics_complete.csv', index=False)
print(f"\n✓ Saved detailed statistics: 'category_statistics_complete.csv'")

# ============================================================================
# STEP 2: VISUALIZE ALL CATEGORIES
# ============================================================================

print("\n" + "="*80)
print("STEP 2: VISUALIZATIONS")
print("="*80)

# 1. Product count by category
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Product count
stats_df.plot(x='category', y='n_products', kind='barh', ax=axes[0,0], color='steelblue', legend=False)
axes[0,0].set_title('Number of Products by Category', fontweight='bold', fontsize=12)
axes[0,0].set_xlabel('Number of Products')
axes[0,0].set_ylabel('')

# Average price
stats_df.plot(x='category', y='avg_price', kind='barh', ax=axes[0,1], color='green', legend=False)
axes[0,1].set_title('Average Price by Category', fontweight='bold', fontsize=12)
axes[0,1].set_xlabel('Average Price (₹)')
axes[0,1].set_ylabel('')

# Number of brands
stats_df.plot(x='category', y='n_brands', kind='barh', ax=axes[1,0], color='orange', legend=False)
axes[1,0].set_title('Number of Brands by Category', fontweight='bold', fontsize=12)
axes[1,0].set_xlabel('Number of Brands')
axes[1,0].set_ylabel('')

# Average rating
stats_df.plot(x='category', y='avg_rating', kind='barh', ax=axes[1,1], color='purple', legend=False)
axes[1,1].set_title('Average Rating by Category', fontweight='bold', fontsize=12)
axes[1,1].set_xlabel('Average Rating (0-5)')
axes[1,1].set_ylabel('')
axes[1,1].set_xlim(0, 5)

plt.tight_layout()
plt.savefig('all_categories_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 'all_categories_overview.png'")

# 2. Price distribution comparison for all categories
n_cats = len(all_categories)
n_cols = 3
n_rows = (n_cats + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes = axes.flatten() if n_cats > 1 else [axes]

for idx, cat in enumerate(all_categories):
    cat_df = df[df['category'] == cat]
    prices = cat_df['sale_price']
    
    # Remove extreme outliers for visualization
    price_99 = prices.quantile(0.99)
    prices_viz = prices[prices <= price_99]
    
    axes[idx].hist(prices_viz, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{cat}\n(n={len(cat_df):,}, median=₹{prices.median():.0f})', fontsize=10)
    axes[idx].set_xlabel('Price (₹)', fontsize=9)
    axes[idx].set_ylabel('Frequency', fontsize=9)
    axes[idx].axvline(prices.median(), color='red', linestyle='--', linewidth=2)

# Hide extra subplots
for idx in range(len(all_categories), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('all_categories_price_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 'all_categories_price_distributions.png'")

# ============================================================================
# STEP 3: CALCULATE PAIRWISE SIMILARITY FOR ALL CATEGORIES
# ============================================================================

print("\n" + "="*80)
print("STEP 3: CALCULATING PAIRWISE SIMILARITY")
print("="*80)

def calculate_similarity(df, cat1, cat2):
    """Calculate comprehensive similarity between two categories"""
    
    cat1_df = df[df['category'] == cat1].copy()
    cat2_df = df[df['category'] == cat2].copy()
    
    # 1. Price similarity
    price1 = cat1_df['sale_price'].mean()
    price2 = cat2_df['sale_price'].mean()
    price_similarity = 1 - abs(price1 - price2) / max(price1, price2)
    
    # 2. Brand overlap
    brands1 = set(cat1_df['brand'].dropna().unique())
    brands2 = set(cat2_df['brand'].dropna().unique())
    brand_overlap = len(brands1 & brands2) / max(len(brands1), 1)
    
    # 3. Size similarity
    size_ratio = min(len(cat1_df), len(cat2_df)) / max(len(cat1_df), len(cat2_df))
    
    # 4. Price variance similarity
    std1 = cat1_df['sale_price'].std()
    std2 = cat2_df['sale_price'].std()
    std_similarity = 1 - abs(std1 - std2) / max(std1, std2)
    
    # 5. Rating similarity
    rating1 = pd.to_numeric(cat1_df['rating'], errors='coerce').mean()
    rating2 = pd.to_numeric(cat2_df['rating'], errors='coerce').mean()
    if not np.isnan(rating1) and not np.isnan(rating2):
        rating_similarity = 1 - abs(rating1 - rating2) / 5.0
    else:
        rating_similarity = 0.5
    
    # Composite score with weights
    composite = (
        price_similarity * 0.30 +      # Price is important
        brand_overlap * 0.25 +          # Brand overlap indicates similarity
        size_ratio * 0.10 +             # Size shouldn't matter too much
        std_similarity * 0.20 +         # Price variance similarity
        rating_similarity * 0.15        # Rating similarity
    )
    
    return {
        'cat1': cat1,
        'cat2': cat2,
        'composite_score': composite,
        'price_similarity': price_similarity,
        'brand_overlap': brand_overlap,
        'size_ratio': size_ratio,
        'std_similarity': std_similarity,
        'rating_similarity': rating_similarity,
        'n1': len(cat1_df),
        'n2': len(cat2_df),
        'price1': price1,
        'price2': price2,
        'price_diff_pct': abs(price1 - price2) / max(price1, price2) * 100
    }

# Calculate similarity for ALL pairs
print(f"\nCalculating similarity for all {len(all_categories)*(len(all_categories)-1)//2} category pairs...")

all_pairs = []
for cat1, cat2 in combinations(all_categories, 2):
    similarity = calculate_similarity(df, cat1, cat2)
    all_pairs.append(similarity)

pairs_df = pd.DataFrame(all_pairs)
pairs_df = pairs_df.sort_values('composite_score', ascending=False)

# Save all pairs
pairs_df.to_csv('all_category_pairs_similarity.csv', index=False)
print(f"✓ Saved all {len(pairs_df)} pairs: 'all_category_pairs_similarity.csv'")

# ============================================================================
# STEP 4: AUTO-SELECT BEST DOMAIN PAIRS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: AUTO-SELECTING OPTIMAL DOMAIN PAIRS")
print("="*80)

# PAIR 1: MOST SIMILAR (High Transferability Expected)
best_similar = pairs_df.iloc[0]
print(f"\n{'='*60}")
print(f"✅ DOMAIN PAIR 1: HIGH SIMILARITY (Transfer As-Is Expected)")
print(f"{'='*60}")
print(f"Source: {best_similar['cat1']}")
print(f"Target: {best_similar['cat2']}")
print(f"\n📊 Similarity Metrics:")
print(f"  Composite Score: {best_similar['composite_score']:.3f}")
print(f"  Price Similarity: {best_similar['price_similarity']:.3f} (₹{best_similar['price1']:.0f} vs ₹{best_similar['price2']:.0f})")
print(f"  Brand Overlap: {best_similar['brand_overlap']:.3f} ({best_similar['brand_overlap']*100:.1f}%)")
print(f"  Size Ratio: {best_similar['size_ratio']:.3f}")
print(f"  Std Similarity: {best_similar['std_similarity']:.3f}")
print(f"\n🎯 Expected: HIGH transferability - Model should transfer well with minimal changes")

# PAIR 2: MODERATE SIMILARITY (Fine-Tuning Expected)
moderate_pairs = pairs_df[(pairs_df['composite_score'] >= 0.35) & (pairs_df['composite_score'] <= 0.55)]
if len(moderate_pairs) > 0:
    best_moderate = moderate_pairs.iloc[0]
    print(f"\n{'='*60}")
    print(f"⚠️  DOMAIN PAIR 2: MODERATE SIMILARITY (Fine-Tuning Expected)")
    print(f"{'='*60}")
    print(f"Source: {best_moderate['cat1']}")
    print(f"Target: {best_moderate['cat2']}")
    print(f"\n📊 Similarity Metrics:")
    print(f"  Composite Score: {best_moderate['composite_score']:.3f}")
    print(f"  Price Difference: {best_moderate['price_diff_pct']:.1f}%")
    print(f"  Brand Overlap: {best_moderate['brand_overlap']:.3f}")
    print(f"\n🎯 Expected: MODERATE transferability - Fine-tuning recommended")

# PAIR 3: LOW SIMILARITY - Price-based
print(f"\n{'='*60}")
print(f"❌ DOMAIN PAIR 3: LOW SIMILARITY - PREMIUM → BUDGET")
print(f"{'='*60}")
price_75 = df['sale_price'].quantile(0.75)
price_25 = df['sale_price'].quantile(0.25)
print(f"Source: Premium Segment (≥75th percentile, ₹{price_75:.2f})")
print(f"Target: Budget Segment (≤25th percentile, ₹{price_25:.2f})")
print(f"\n🎯 Expected: LOW transferability - Customer price sensitivity differs")

# PAIR 4: BRAND-BASED
brand_counts = df['brand'].value_counts()
n_popular = (brand_counts >= 50).sum()
n_niche = (brand_counts < 10).sum()
print(f"\n{'='*60}")
print(f"❌ DOMAIN PAIR 4: LOW SIMILARITY - POPULAR → NICHE BRANDS")
print(f"{'='*60}")
print(f"Source: Popular Brands ({n_popular} brands with ≥50 products)")
print(f"Target: Niche Brands ({n_niche} brands with <10 products)")
print(f"\n🎯 Expected: LOW-MODERATE transferability - Different brand strategies")

# ============================================================================
# STEP 5: CREATE SIMILARITY HEATMAP
# ============================================================================

print("\n" + "="*80)
print("STEP 5: CREATING SIMILARITY HEATMAP")
print("="*80)

# Create similarity matrix
n_categories = len(all_categories)
similarity_matrix = np.zeros((n_categories, n_categories))
cat_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}

for _, row in pairs_df.iterrows():
    idx1 = cat_to_idx[row['cat1']]
    idx2 = cat_to_idx[row['cat2']]
    score = row['composite_score']
    similarity_matrix[idx1, idx2] = score
    similarity_matrix[idx2, idx1] = score

np.fill_diagonal(similarity_matrix, 1.0)

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    similarity_matrix,
    xticklabels=all_categories,
    yticklabels=all_categories,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Similarity Score'},
    square=True,
    ax=ax
)
ax.set_title('Category Similarity Matrix\n(Green = Similar, Red = Dissimilar)', 
             fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('category_similarity_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 'category_similarity_heatmap.png'")

# ============================================================================
# STEP 6: EXPORT DOMAIN PAIR DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 6: EXPORTING DOMAIN DATA")
print("="*80)

# Domain Pair 1
pair1_source = df[df['category'] == best_similar['cat1']].copy()
pair1_target = df[df['category'] == best_similar['cat2']].copy()
pair1_source.to_csv('domain_pair1_source.csv', index=False)
pair1_target.to_csv('domain_pair1_target.csv', index=False)
print(f"✓ Pair 1: {best_similar['cat1']} → {best_similar['cat2']}")
print(f"  Saved: domain_pair1_source.csv ({len(pair1_source):,} products)")
print(f"  Saved: domain_pair1_target.csv ({len(pair1_target):,} products)")

# Domain Pair 2
if len(moderate_pairs) > 0:
    pair2_source = df[df['category'] == best_moderate['cat1']].copy()
    pair2_target = df[df['category'] == best_moderate['cat2']].copy()
    pair2_source.to_csv('domain_pair2_source.csv', index=False)
    pair2_target.to_csv('domain_pair2_target.csv', index=False)
    print(f"\n✓ Pair 2: {best_moderate['cat1']} → {best_moderate['cat2']}")
    print(f"  Saved: domain_pair2_source.csv ({len(pair2_source):,} products)")
    print(f"  Saved: domain_pair2_target.csv ({len(pair2_target):,} products)")

# Domain Pair 3 (Premium → Budget)
premium = df[df['sale_price'] >= price_75].copy()
budget = df[df['sale_price'] <= price_25].copy()
premium.to_csv('domain_pair3_source.csv', index=False)
budget.to_csv('domain_pair3_target.csv', index=False)
print(f"\n✓ Pair 3: Premium → Budget")
print(f"  Saved: domain_pair3_source.csv ({len(premium):,} products)")
print(f"  Saved: domain_pair3_target.csv ({len(budget):,} products)")

# Domain Pair 4 (Popular → Niche brands)
popular_brands = brand_counts[brand_counts >= 50].index
niche_brands = brand_counts[brand_counts < 10].index
popular = df[df['brand'].isin(popular_brands)].copy()
niche = df[df['brand'].isin(niche_brands)].copy()
popular.to_csv('domain_pair4_source.csv', index=False)
niche.to_csv('domain_pair4_target.csv', index=False)
print(f"\n✓ Pair 4: Popular → Niche Brands")
print(f"  Saved: domain_pair4_source.csv ({len(popular):,} products)")
print(f"  Saved: domain_pair4_target.csv ({len(niche):,} products)")

# ============================================================================
# STEP 7: FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 COMPLETE EDA FINISHED!")
print("="*80)

print("\n📦 Dataset Summary:")
print(f"  Total Products: {len(df):,}")
print(f"  Total Categories: {len(all_categories)}")
print(f"  Category Pairs Analyzed: {len(pairs_df)}")

print("\n✅ Files Created:")
print("  1. category_statistics_complete.csv")
print("  2. all_category_pairs_similarity.csv")
print("  3. all_categories_overview.png")
print("  4. all_categories_price_distributions.png")
print("  5. category_similarity_heatmap.png")
print("  6-13. domain_pair[1-4]_[source/target].csv")

print("\n🔄 Recommended Domain Pairs:")
print(f"  Pair 1 (HIGH): {best_similar['cat1']} → {best_similar['cat2']} (Score: {best_similar['composite_score']:.3f})")
if len(moderate_pairs) > 0:
    print(f"  Pair 2 (MOD):  {best_moderate['cat1']} → {best_moderate['cat2']} (Score: {best_moderate['composite_score']:.3f})")
print(f"  Pair 3 (LOW):  Premium → Budget")
print(f"  Pair 4 (LOW):  Popular → Niche Brands")

print("\n🎯 Next Steps:")
print("  1. Review the similarity heatmap")
print("  2. Check if recommended pairs make business sense")
print("  3. Proceed to Week 2: Synthetic customer generation & RFM")
print("  4. Build baseline segmentation models")

print("\n" + "="*80)