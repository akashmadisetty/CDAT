"""
Create individual domain CSV files for specific categories
Extracts each category from BigBasket_v3.csv into separate files

Author: Transfer Learning Framework Team
Date: November 6, 2025
"""

import pandas as pd
import os

print("="*80)
print("CREATING DOMAIN CSV FILES BY CATEGORY")
print("="*80)

# Load BigBasket_v3.csv
print("\n📂 Loading BigBasket_v3.csv...")
df = pd.read_csv('data/processed/BigBasket_v3.csv')
print(f"✓ Loaded {len(df):,} products")

# Define categories to extract
target_categories = [
    'Beverages',
    'Gourmet & World Food'
]

print(f"\n📋 Target categories: {len(target_categories)}")
for cat in target_categories:
    print(f"  • {cat}")

# Create output directory if it doesn't exist
output_dir = 'data/domains'
os.makedirs(output_dir, exist_ok=True)
print(f"\n📁 Output directory: {output_dir}/")

# Extract and save each category
print("\n" + "="*80)
print("EXTRACTING CATEGORIES")
print("="*80)

extracted_count = 0
missing_categories = []

for category in target_categories:
    # Filter data for this category
    category_df = df[df['category'] == category]
    
    if len(category_df) == 0:
        print(f"\n⚠️  '{category}' - NOT FOUND")
        missing_categories.append(category)
        continue
    
    # Create filename (replace special characters)
    filename = category.replace(', ', '_').replace(' ', '_').replace('&', 'and')
    filename = f"domain_{filename}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save to CSV
    category_df.to_csv(filepath, index=False)
    
    # Statistics
    avg_price = category_df['sale_price'].mean()
    avg_rating = category_df['rating_clean'].mean()
    n_brands = category_df['brand'].nunique() if 'brand' in category_df.columns else 0
    
    print(f"\n✅ {category}")
    print(f"   File: {filename}")
    print(f"   Products: {len(category_df):,}")
    print(f"   Avg Price: ₹{avg_price:.2f}")
    print(f"   Avg Rating: {avg_rating:.2f}")
    print(f"   Brands: {n_brands}")
    
    extracted_count += 1

# Handle missing categories
if missing_categories:
    print("\n" + "="*80)
    print("⚠️  MISSING CATEGORIES")
    print("="*80)
    print("\nThe following categories were not found in BigBasket_v3.csv:")
    for cat in missing_categories:
        print(f"  ❌ {cat}")
    
    print("\nAvailable categories in BigBasket_v3.csv:")
    available_cats = df['category'].unique()
    for cat in sorted(available_cats):
        print(f"  ✓ {cat}")
    
    print("\n💡 Tip: Check for exact spelling, capitalization, and spacing.")

# Summary
print("\n" + "="*80)
print("✅ DOMAIN FILES CREATED!")
print("="*80)

print(f"\n📊 Summary:")
print(f"  Target categories: {len(target_categories)}")
print(f"  Successfully extracted: {extracted_count}")
print(f"  Missing/Not found: {len(missing_categories)}")
print(f"  Total products in v3: {len(df):,}")

print(f"\n📁 Output files in: {output_dir}/")
if extracted_count > 0:
    print("\nGenerated files:")
    for category in target_categories:
        if category not in missing_categories:
            filename = category.replace(', ', '_').replace(' ', '_').replace('&', 'and')
            filename = f"domain_{filename}.csv"
            print(f"  ✓ {filename}")

print("\n🎯 Ready to use for domain-specific analysis!")
