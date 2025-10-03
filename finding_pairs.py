# Find Better Domain Pairs for Transfer Learning
# Goal: Replace Domain Pair 1 with a truly SIMILAR pair

import pandas as pd
import numpy as np
from itertools import combinations

print("="*80)
print("FINDING OPTIMAL DOMAIN PAIRS FOR TRANSFER LEARNING")
print("="*80)

# Load the BigBasket data
df = pd.read_csv('data/BigBasket.csv')

# Get all categories
categories = df['category'].unique()
print(f"\n📦 Available Categories ({len(categories)}):")
for i, cat in enumerate(categories, 1):
    count = len(df[df['category'] == cat])
    avg_price = df[df['category'] == cat]['sale_price'].mean()
    print(f"  {i:2d}. {cat:40s} ({count:5,} products, Avg: ₹{avg_price:7,.2f})")

# Function to calculate similarity between two categories
def calculate_category_similarity(df, cat1, cat2):
    """Calculate comprehensive similarity between two categories"""
    
    cat1_df = df[df['category'] == cat1].copy()
    cat2_df = df[df['category'] == cat2].copy()
    
    # 1. Price similarity (0-1, higher is more similar)
    price1 = cat1_df['sale_price'].mean()
    price2 = cat2_df['sale_price'].mean()
    price_similarity = 1 - abs(price1 - price2) / max(price1, price2)
    
    # 2. Brand overlap (0-1)
    brands1 = set(cat1_df['brand'].dropna().unique())
    brands2 = set(cat2_df['brand'].dropna().unique())
    if len(brands1) > 0:
        brand_overlap = len(brands1 & brands2) / len(brands1)
    else:
        brand_overlap = 0
    
    # 3. Size similarity (penalize very different sizes)
    size_ratio = min(len(cat1_df), len(cat2_df)) / max(len(cat1_df), len(cat2_df))
    
    # 4. Rating similarity (if available)
    rating1 = pd.to_numeric(cat1_df['rating'], errors='coerce').mean()
    rating2 = pd.to_numeric(cat2_df['rating'], errors='coerce').mean()
    if not np.isnan(rating1) and not np.isnan(rating2):
        rating_similarity = 1 - abs(rating1 - rating2) / 5.0  # ratings are 0-5
    else:
        rating_similarity = 0.5  # neutral if no ratings
    
    # Composite similarity score
    composite = (
        price_similarity * 0.35 +
        brand_overlap * 0.25 +
        size_ratio * 0.15 +
        rating_similarity * 0.25
    )
    
    return {
        'cat1': cat1,
        'cat2': cat2,
        'price_similarity': price_similarity,
        'brand_overlap': brand_overlap,
        'size_ratio': size_ratio,
        'rating_similarity': rating_similarity,
        'composite_score': composite,
        'size1': len(cat1_df),
        'size2': len(cat2_df),
        'price1': price1,
        'price2': price2
    }

# Calculate similarity for ALL category pairs
print("\n" + "="*80)
print("CALCULATING SIMILARITY FOR ALL CATEGORY PAIRS")
print("="*80)

all_pairs = []
for cat1, cat2 in combinations(categories, 2):
    similarity = calculate_category_similarity(df, cat1, cat2)
    all_pairs.append(similarity)

# Convert to DataFrame for easy analysis
pairs_df = pd.DataFrame(all_pairs)
pairs_df = pairs_df.sort_values('composite_score', ascending=False)

print("\n🔵 TOP 10 MOST SIMILAR CATEGORY PAIRS (for Domain Pair 1):")
print("="*80)
for idx, row in pairs_df.head(10).iterrows():
    print(f"\n{row['cat1']} ↔ {row['cat2']}")
    print(f"  Composite Score: {row['composite_score']:.3f}")
    print(f"  Price Similarity: {row['price_similarity']:.3f} (₹{row['price1']:.0f} vs ₹{row['price2']:.0f})")
    print(f"  Brand Overlap: {row['brand_overlap']:.3f}")
    print(f"  Size Ratio: {row['size_ratio']:.3f} ({row['size1']:,} vs {row['size2']:,})")
    print(f"  ✓ Expected: HIGH transferability")

print("\n" + "="*80)
print("\n🟡 MODERATELY SIMILAR PAIRS (for Domain Pair 2):")
print("="*80)
moderate_pairs = pairs_df[
    (pairs_df['composite_score'] >= 0.35) & 
    (pairs_df['composite_score'] <= 0.55)
].head(10)

for idx, row in moderate_pairs.iterrows():
    print(f"\n{row['cat1']} ↔ {row['cat2']}")
    print(f"  Composite Score: {row['composite_score']:.3f}")
    print(f"  Price Similarity: {row['price_similarity']:.3f}")
    print(f"  Brand Overlap: {row['brand_overlap']:.3f}")
    print(f"  ⚠️  Expected: MODERATE transferability")

print("\n" + "="*80)
print("\n🔴 LEAST SIMILAR PAIRS (for validation):")
print("="*80)
for idx, row in pairs_df.tail(5).iterrows():
    print(f"\n{row['cat1']} ↔ {row['cat2']}")
    print(f"  Composite Score: {row['composite_score']:.3f}")
    print(f"  Price Difference: {abs(row['price1'] - row['price2']):.0f} ({abs(1-row['price1']/row['price2'])*100:.0f}%)")
    print(f"  Brand Overlap: {row['brand_overlap']:.3f}")
    print(f"  ❌ Expected: LOW transferability")

# ============================================================================
# RECOMMENDED DOMAIN PAIRS BASED ON ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("🎯 RECOMMENDED DOMAIN PAIRS FOR YOUR PROJECT")
print("="*80)

# Get top similar pair
best_similar = pairs_df.iloc[0]
print(f"\n✅ DOMAIN PAIR 1 (SIMILAR - High Transferability Expected):")
print(f"   Source: {best_similar['cat1']}")
print(f"   Target: {best_similar['cat2']}")
print(f"   Similarity Score: {best_similar['composite_score']:.3f}")
print(f"   Why: Similar prices (₹{best_similar['price1']:.0f} vs ₹{best_similar['price2']:.0f}), {best_similar['brand_overlap']*100:.1f}% brand overlap")

# Get moderate pair
if len(moderate_pairs) > 0:
    best_moderate = moderate_pairs.iloc[0]
    print(f"\n⚠️  DOMAIN PAIR 2 (MODERATE - Fine-tuning Expected):")
    print(f"   Source: {best_moderate['cat1']}")
    print(f"   Target: {best_moderate['cat2']}")
    print(f"   Similarity Score: {best_moderate['composite_score']:.3f}")
    print(f"   Why: Some similarities but noticeable differences")

# Keep your existing Premium → Budget pair
print(f"\n❌ DOMAIN PAIR 3 (LOW - Premium → Budget):")
print(f"   Source: Premium Segment (>75th percentile)")
print(f"   Target: Budget Segment (<25th percentile)")
print(f"   Why: Price-based segmentation with different customer behaviors")

# Keep your existing Popular → Niche pair
print(f"\n❌ DOMAIN PAIR 4 (LOW - Popular → Niche Brands):")
print(f"   Source: Popular Brands (≥50 products)")
print(f"   Target: Niche Brands (<10 products)")
print(f"   Why: Brand popularity-based segmentation")

# ============================================================================
# VISUALIZE SIMILARITY MATRIX
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

# Create similarity matrix
n_categories = len(categories)
similarity_matrix = np.zeros((n_categories, n_categories))

cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

for _, row in pairs_df.iterrows():
    idx1 = cat_to_idx[row['cat1']]
    idx2 = cat_to_idx[row['cat2']]
    score = row['composite_score']
    similarity_matrix[idx1, idx2] = score
    similarity_matrix[idx2, idx1] = score  # Symmetric

# Set diagonal to 1 (category similar to itself)
np.fill_diagonal(similarity_matrix, 1.0)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    similarity_matrix,
    xticklabels=categories,
    yticklabels=categories,
    annot=False,
    cmap='RdYlGn',
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Similarity Score'}
)
plt.title('Category Similarity Matrix\n(Green = Similar, Red = Dissimilar)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('category_similarity_matrix.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Similarity matrix saved as 'category_similarity_matrix.png'")

# ============================================================================
# EXPORT RECOMMENDED PAIRS
# ============================================================================

recommended_pairs = [
    {
        'pair_id': 1,
        'type': 'SIMILAR',
        'source': best_similar['cat1'],
        'target': best_similar['cat2'],
        'expected_transferability': 'HIGH',
        'similarity_score': best_similar['composite_score']
    },
    {
        'pair_id': 2,
        'type': 'MODERATE',
        'source': best_moderate['cat1'] if len(moderate_pairs) > 0 else 'Beauty & Hygiene',
        'target': best_moderate['cat2'] if len(moderate_pairs) > 0 else 'Baby Care',
        'expected_transferability': 'MODERATE',
        'similarity_score': best_moderate['composite_score'] if len(moderate_pairs) > 0 else 0.41
    },
    {
        'pair_id': 3,
        'type': 'PRICE_SEGMENT',
        'source': 'Premium Segment',
        'target': 'Budget Segment',
        'expected_transferability': 'LOW',
        'similarity_score': 0.19
    },
    {
        'pair_id': 4,
        'type': 'BRAND_POPULARITY',
        'source': 'Popular Brands',
        'target': 'Niche Brands',
        'expected_transferability': 'MODERATE-LOW',
        'similarity_score': 0.40
    }
]

recommended_df = pd.DataFrame(recommended_pairs)
recommended_df.to_csv('recommended_domain_pairs.csv', index=False)
print(f"✓ Recommended pairs saved as 'recommended_domain_pairs.csv'")

print("\n" + "="*80)
print("🎉 ANALYSIS COMPLETE!")
print("="*80)
print("\n📋 NEXT STEPS:")
print("  1. Review the recommended domain pairs above")
print("  2. Update your domain pair definitions in Week 2 code")
print("  3. Use the similarity matrix to understand relationships")
print("  4. Proceed with synthetic customer generation using these pairs")
print("\n" + "="*80)