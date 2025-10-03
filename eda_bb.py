# BigBasket Exploratory Data Analysis (EDA)
# Week 1: Day 3-4
# Goal: Understand data and define 4 domain pairs for transfer learning experiments

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("="*80)
print("STEP 1: LOADING BIGBASKET DATA")
print("="*80)

# Load the CSV file
df = pd.read_csv('data/BigBasket.csv')  # Update path as needed

print(f"\n✓ Data loaded successfully!")
print(f"✓ Total products: {len(df):,}")
print(f"✓ Total columns: {len(df.columns)}")

# ============================================================================
# STEP 2: BASIC OVERVIEW
# ============================================================================

print("\n" + "="*80)
print("STEP 2: BASIC DATA OVERVIEW")
print("="*80)

print("\n📋 Column Names:")
print(df.columns.tolist())

print("\n📊 First 5 rows:")
print(df.head())

print("\n📈 Data types:")
print(df.dtypes)

print("\n📉 Basic statistics:")
print(df.describe())

print("\n🔍 Dataset shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ============================================================================
# STEP 3: MISSING VALUES ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: MISSING VALUES ANALYSIS")
print("="*80)

missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_data) > 0:
    print("\n⚠️  Columns with missing values:")
    print(missing_data.to_string(index=False))
    
    # Visualize missing data
    plt.figure(figsize=(10, 6))
    plt.barh(missing_data['Column'], missing_data['Missing_Percentage'])
    plt.xlabel('Missing Percentage (%)')
    plt.title('Missing Values by Column')
    plt.tight_layout()
    plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
    print("\n✓ Missing values chart saved as 'missing_values.png'")
else:
    print("\n✓ No missing values found!")

# ============================================================================
# STEP 4: CATEGORY ANALYSIS (MAIN FOCUS)
# ============================================================================

print("\n" + "="*80)
print("STEP 4: PRODUCT CATEGORY ANALYSIS")
print("="*80)

# Main categories
print("\n📦 Main Categories:")
category_counts = df['category'].value_counts()
print(category_counts)

print("\n📊 Category Distribution:")
for cat, count in category_counts.items():
    percentage = (count / len(df) * 100)
    print(f"  {cat}: {count:,} products ({percentage:.1f}%)")

# Visualize category distribution
plt.figure(figsize=(12, 6))
category_counts.plot(kind='bar', color='steelblue')
plt.title('Product Distribution by Category', fontsize=14, fontweight='bold')
plt.xlabel('Category')
plt.ylabel('Number of Products')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Category distribution chart saved as 'category_distribution.png'")

# ============================================================================
# STEP 5: DETAILED ANALYSIS FOR EACH DOMAIN
# ============================================================================

print("\n" + "="*80)
print("STEP 5: DETAILED DOMAIN ANALYSIS")
print("="*80)

# Function to analyze a category
def analyze_category(df, category_name):
    """Analyze a specific product category"""
    cat_df = df[df['category'] == category_name].copy()
    
    print(f"\n{'='*60}")
    print(f"📦 {category_name.upper()}")
    print(f"{'='*60}")
    
    # Basic stats
    print(f"\n📊 Basic Statistics:")
    print(f"  Total Products: {len(cat_df):,}")
    print(f"  Percentage of Dataset: {len(cat_df)/len(df)*100:.2f}%")
    
    # Price analysis
    if 'sale_price' in cat_df.columns:
        print(f"\n💰 Price Statistics:")
        print(f"  Average Price: ₹{cat_df['sale_price'].mean():.2f}")
        print(f"  Median Price: ₹{cat_df['sale_price'].median():.2f}")
        print(f"  Min Price: ₹{cat_df['sale_price'].min():.2f}")
        print(f"  Max Price: ₹{cat_df['sale_price'].max():.2f}")
        print(f"  Std Dev: ₹{cat_df['sale_price'].std():.2f}")
    
    # Rating analysis
    if 'rating' in cat_df.columns:
        # Clean ratings (convert to numeric, handle missing)
        cat_df['rating_clean'] = pd.to_numeric(cat_df['rating'], errors='coerce')
        valid_ratings = cat_df['rating_clean'].dropna()
        
        if len(valid_ratings) > 0:
            print(f"\n⭐ Rating Statistics:")
            print(f"  Average Rating: {valid_ratings.mean():.2f}")
            print(f"  Median Rating: {valid_ratings.median():.2f}")
            print(f"  Products with Ratings: {len(valid_ratings):,} ({len(valid_ratings)/len(cat_df)*100:.1f}%)")
    
    # Brand analysis
    if 'brand' in cat_df.columns:
        brand_counts = cat_df['brand'].value_counts()
        print(f"\n🏷️  Brand Statistics:")
        print(f"  Unique Brands: {len(brand_counts)}")
        print(f"  Top 5 Brands:")
        for i, (brand, count) in enumerate(brand_counts.head(5).items(), 1):
            print(f"    {i}. {brand}: {count} products")
    
    # Sub-category analysis
    if 'sub_category' in cat_df.columns:
        subcat_counts = cat_df['sub_category'].value_counts()
        print(f"\n📑 Sub-categories: {len(subcat_counts)}")
        print(f"  Top 5 Sub-categories:")
        for i, (subcat, count) in enumerate(subcat_counts.head(5).items(), 1):
            print(f"    {i}. {subcat}: {count} products")
    
    return cat_df

# Analyze key categories for our domain pairs
categories_to_analyze = [
    'Beauty & Hygiene',
    'Fruits & Vegetables',  # If exists, or use closest grocery category
    'Baby Care',
    'Beverages',
    'Gourmet & World Food',
    'Kitchen, Garden & Pets',
    'Cleaning & Household'
]

category_data = {}
for cat in categories_to_analyze:
    if cat in df['category'].unique():
        category_data[cat] = analyze_category(df, cat)
    else:
        # Try to find closest match
        similar = [c for c in df['category'].unique() if cat.lower() in c.lower() or c.lower() in cat.lower()]
        if similar:
            print(f"\n⚠️  '{cat}' not found. Using '{similar[0]}' instead.")
            category_data[similar[0]] = analyze_category(df, similar[0])

# ============================================================================
# STEP 6: PRICE DISTRIBUTION COMPARISON
# ============================================================================

print("\n" + "="*80)
print("STEP 6: PRICE DISTRIBUTION COMPARISON")
print("="*80)

# Create price comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Price Distribution by Category', fontsize=16, fontweight='bold')

categories_for_viz = list(category_data.keys())[:4]  # Take first 4 categories

for idx, cat in enumerate(categories_for_viz):
    ax = axes[idx // 2, idx % 2]
    cat_df = category_data[cat]
    
    if 'sale_price' in cat_df.columns:
        # Remove outliers for better visualization (above 99th percentile)
        price_99 = cat_df['sale_price'].quantile(0.99)
        prices = cat_df['sale_price'][cat_df['sale_price'] <= price_99]
        
        ax.hist(prices, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title(f'{cat}\n(n={len(cat_df):,} products)', fontweight='bold')
        ax.set_xlabel('Price (₹)')
        ax.set_ylabel('Frequency')
        ax.axvline(prices.median(), color='red', linestyle='--', label=f'Median: ₹{prices.median():.0f}')
        ax.legend()

plt.tight_layout()
plt.savefig('price_distributions.png', dpi=300, bbox_inches='tight')
print("\n✓ Price distribution chart saved as 'price_distributions.png'")

# ============================================================================
# STEP 7: DEFINE DOMAIN PAIRS (CRITICAL FOR TRANSFER LEARNING)
# ============================================================================

print("\n" + "="*80)
print("STEP 7: DEFINING DOMAIN PAIRS FOR TRANSFER LEARNING")
print("="*80)

# Helper function to compare two domains
def compare_domains(source_df, target_df, source_name, target_name):
    """Compare two domains for transferability analysis"""
    print(f"\n{'='*60}")
    print(f"🔄 Domain Pair: {source_name} → {target_name}")
    print(f"{'='*60}")
    
    print(f"\n📊 Size Comparison:")
    print(f"  Source ({source_name}): {len(source_df):,} products")
    print(f"  Target ({target_name}): {len(target_df):,} products")
    print(f"  Size Ratio: {len(target_df)/len(source_df):.2f}x")
    
    # Price comparison
    if 'sale_price' in source_df.columns and 'sale_price' in target_df.columns:
        print(f"\n💰 Price Comparison:")
        print(f"  Source Avg Price: ₹{source_df['sale_price'].mean():.2f}")
        print(f"  Target Avg Price: ₹{target_df['sale_price'].mean():.2f}")
        print(f"  Price Difference: {abs(source_df['sale_price'].mean() - target_df['sale_price'].mean()):.2f} ({abs(1 - target_df['sale_price'].mean()/source_df['sale_price'].mean())*100:.1f}%)")
    
    # Brand overlap
    if 'brand' in source_df.columns and 'brand' in target_df.columns:
        source_brands = set(source_df['brand'].dropna().unique())
        target_brands = set(target_df['brand'].dropna().unique())
        common_brands = source_brands & target_brands
        
        print(f"\n🏷️  Brand Overlap:")
        print(f"  Source Brands: {len(source_brands)}")
        print(f"  Target Brands: {len(target_brands)}")
        print(f"  Common Brands: {len(common_brands)}")
        print(f"  Overlap Percentage: {len(common_brands)/len(source_brands)*100:.1f}%")
    
    # Initial transferability assessment
    print(f"\n🎯 Initial Transferability Assessment:")
    
    # Calculate simple similarity score based on price similarity
    if 'sale_price' in source_df.columns and 'sale_price' in target_df.columns:
        price_similarity = 1 - abs(source_df['sale_price'].mean() - target_df['sale_price'].mean()) / max(source_df['sale_price'].mean(), target_df['sale_price'].mean())
        brand_similarity = len(common_brands) / len(source_brands) if 'brand' in source_df.columns else 0
        
        rough_score = (price_similarity * 0.5 + brand_similarity * 0.5)
        
        print(f"  Price Similarity: {price_similarity:.2f}")
        print(f"  Brand Similarity: {brand_similarity:.2f}")
        print(f"  Rough Transferability Score: {rough_score:.2f}")
        
        if rough_score > 0.6:
            print("  ✓ Expected: HIGH transferability (transfer as-is might work)")
        elif rough_score > 0.3:
            print("  ⚠️  Expected: MODERATE transferability (fine-tuning recommended)")
        else:
            print("  ❌ Expected: LOW transferability (train new model)")

# DOMAIN PAIR 1: Similar categories (should have HIGH transferability)
print("\n" + "🔵"*30)
print("DOMAIN PAIR 1: PRODUCT CATEGORY TRANSFER (SIMILAR)")
print("🔵"*30)

# Find two similar categories from what we have
available_cats = list(category_data.keys())
if len(available_cats) >= 2:
    source_cat1 = available_cats[0]
    target_cat1 = available_cats[1]
    
    compare_domains(
        category_data[source_cat1],
        category_data[target_cat1],
        source_cat1,
        target_cat1
    )

# DOMAIN PAIR 2: Different categories (should have MODERATE transferability)
print("\n" + "🟡"*30)
print("DOMAIN PAIR 2: PRODUCT CATEGORY TRANSFER (DIFFERENT)")
print("🟡"*30)

if len(available_cats) >= 3:
    source_cat2 = available_cats[0]
    target_cat2 = available_cats[2]
    
    compare_domains(
        category_data[source_cat2],
        category_data[target_cat2],
        source_cat2,
        target_cat2
    )

# DOMAIN PAIR 3: Premium vs Budget (price-based segmentation)
print("\n" + "🟠"*30)
print("DOMAIN PAIR 3: PREMIUM → BUDGET SEGMENT")
print("🟠"*30)

if 'sale_price' in df.columns:
    # Calculate price percentiles
    price_75 = df['sale_price'].quantile(0.75)
    price_25 = df['sale_price'].quantile(0.25)
    
    premium_products = df[df['sale_price'] >= price_75].copy()
    budget_products = df[df['sale_price'] <= price_25].copy()
    
    print(f"\n💎 Premium Segment (≥75th percentile, ₹{price_75:.2f}):")
    print(f"  Products: {len(premium_products):,}")
    print(f"  Avg Price: ₹{premium_products['sale_price'].mean():.2f}")
    
    print(f"\n💵 Budget Segment (≤25th percentile, ₹{price_25:.2f}):")
    print(f"  Products: {len(budget_products):,}")
    print(f"  Avg Price: ₹{budget_products['sale_price'].mean():.2f}")
    
    compare_domains(
        premium_products,
        budget_products,
        "Premium Segment",
        "Budget Segment"
    )

# DOMAIN PAIR 4: Popular vs Niche brands
print("\n" + "🔴"*30)
print("DOMAIN PAIR 4: POPULAR BRANDS → NICHE BRANDS")
print("🔴"*30)

if 'brand' in df.columns:
    brand_counts = df['brand'].value_counts()
    
    # Popular brands: >50 products
    popular_brands = brand_counts[brand_counts >= 50].index.tolist()
    # Niche brands: <10 products
    niche_brands = brand_counts[brand_counts < 10].index.tolist()
    
    popular_products = df[df['brand'].isin(popular_brands)].copy()
    niche_products = df[df['brand'].isin(niche_brands)].copy()
    
    print(f"\n🌟 Popular Brands (≥50 products each):")
    print(f"  Number of Brands: {len(popular_brands)}")
    print(f"  Total Products: {len(popular_products):,}")
    print(f"  Top 5 Popular Brands: {popular_brands[:5]}")
    
    print(f"\n🎯 Niche Brands (<10 products each):")
    print(f"  Number of Brands: {len(niche_brands)}")
    print(f"  Total Products: {len(niche_products):,}")
    
    compare_domains(
        popular_products,
        niche_products,
        "Popular Brands",
        "Niche Brands"
    )

# ============================================================================
# STEP 8: OUTLIER DETECTION
# ============================================================================

print("\n" + "="*80)
print("STEP 8: OUTLIER DETECTION")
print("="*80)

if 'sale_price' in df.columns:
    # Price outliers using IQR method
    Q1 = df['sale_price'].quantile(0.25)
    Q3 = df['sale_price'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['sale_price'] < lower_bound) | (df['sale_price'] > upper_bound)]
    
    print(f"\n📊 Price Outlier Analysis:")
    print(f"  Q1 (25th percentile): ₹{Q1:.2f}")
    print(f"  Q3 (75th percentile): ₹{Q3:.2f}")
    print(f"  IQR: ₹{IQR:.2f}")
    print(f"  Lower Bound: ₹{lower_bound:.2f}")
    print(f"  Upper Bound: ₹{upper_bound:.2f}")
    print(f"  Number of Outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")
    
    if len(outliers) > 0:
        print(f"\n🔍 Top 5 Most Expensive Products (Potential Outliers):")
        expensive = df.nlargest(5, 'sale_price')[['product', 'category', 'brand', 'sale_price']]
        print(expensive.to_string(index=False))

# ============================================================================
# STEP 9: SUMMARY & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 9: SUMMARY & NEXT STEPS")
print("="*80)

print("\n✅ EDA COMPLETE! Here's what we found:")
print("\n📦 Dataset Overview:")
print(f"  • Total Products: {len(df):,}")
print(f"  • Categories: {df['category'].nunique()}")
print(f"  • Brands: {df['brand'].nunique() if 'brand' in df.columns else 'N/A'}")
print(f"  • Price Range: ₹{df['sale_price'].min():.2f} - ₹{df['sale_price'].max():.2f}")

print("\n🔄 Defined Domain Pairs for Transfer Learning:")
print("  1. Category Transfer (Similar): Good transferability expected")
print("  2. Category Transfer (Different): Moderate transferability expected")
print("  3. Premium → Budget: Low-moderate transferability expected")
print("  4. Popular → Niche Brands: Low transferability expected")

print("\n📋 Data Quality:")
if len(missing_data) > 0:
    print(f"  ⚠️  Missing values found in {len(missing_data)} columns")
    print("  → Recommendation: Handle missing values in preprocessing")
else:
    print("  ✓ No missing values - data is clean!")

print(f"\n  ⚠️  Outliers: {len(outliers):,} products ({len(outliers)/len(df)*100:.1f}%)")
print("  → Recommendation: Consider removing extreme outliers for modeling")

print("\n🎯 NEXT STEPS (Week 1, Day 5-7):")
print("  1. Preprocess data (handle missing values, normalize prices)")
print("  2. Generate synthetic customer transactions")
print("  3. Create RFM features for each domain")
print("  4. Begin baseline model training (Week 2)")

print("\n✅ Save this analysis and move to UK Retail dataset!")
print("="*80)

# Save processed dataframes for later use
print("\n💾 Saving processed domain data...")
for cat_name, cat_df in category_data.items():
    filename = f"domain_{cat_name.replace(' ', '_').replace('&', 'and').lower()}.csv"
    cat_df.to_csv(filename, index=False)
    print(f"  ✓ Saved {filename}")

if 'premium_products' in locals():
    premium_products.to_csv('domain_premium_segment.csv', index=False)
    budget_products.to_csv('domain_budget_segment.csv', index=False)
    print(f"  ✓ Saved domain_premium_segment.csv")
    print(f"  ✓ Saved domain_budget_segment.csv")

if 'popular_products' in locals():
    popular_products.to_csv('domain_popular_brands.csv', index=False)
    niche_products.to_csv('domain_niche_brands.csv', index=False)
    print(f"  ✓ Saved domain_popular_brands.csv")
    print(f"  ✓ Saved domain_niche_brands.csv")

print("\n🎉 EDA COMPLETE! All domain data saved for Week 2 modeling.")
print("="*80)