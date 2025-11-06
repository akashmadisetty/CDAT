"""
Create BigBasket_v3.csv by merging:
- BB_eggs_meat_fish_processed.csv (for Eggs, Meat & Fish category)
- BigBasket_v2.csv (for all other categories except Fruits & Vegetables)

Author: Transfer Learning Framework Team
Date: November 6, 2025
"""

import pandas as pd
import numpy as np

print("="*80)
print("CREATING BIGBASKET_V3.CSV")
print("Merging processed Eggs/Meat/Fish data with other categories")
print("="*80)

# Step 1: Load BigBasket_v2.csv
print("\n📂 Step 1: Loading BigBasket_v2.csv...")
df_v2 = pd.read_csv('data/processed/BigBasket_v2.csv')
print(f"✓ Loaded {len(df_v2):,} rows from BigBasket_v2.csv")

# Show category distribution
print("\n📊 Original category distribution:")
category_counts = df_v2['category'].value_counts()
for cat, count in category_counts.items():
    print(f"  {cat:<40} {count:>6,} rows")

# Step 2: Remove Eggs, Meat & Fish category
print("\n🗑️  Step 2: Removing 'Eggs, Meat & Fish' category...")
df_without_eggs = df_v2[df_v2['category'] != 'Eggs, Meat & Fish'].copy()
print(f"✓ Removed {len(df_v2) - len(df_without_eggs):,} rows")
print(f"  Remaining: {len(df_without_eggs):,} rows")

# Step 3: Remove Fruits & Vegetables category
print("\n🗑️  Step 3: Removing 'Fruits & Vegetables' category...")
df_filtered = df_without_eggs[df_without_eggs['category'] != 'Fruits & Vegetables'].copy()
print(f"✓ Removed {len(df_without_eggs) - len(df_filtered):,} rows")
print(f"  Remaining: {len(df_filtered):,} rows")

# Step 4: Load processed Eggs, Meat & Fish data
print("\n📂 Step 4: Loading BB_eggs_meat_fish_processed.csv...")
df_eggs = pd.read_csv('BB_eggs_meat_fish_processed.csv')
print(f"✓ Loaded {len(df_eggs):,} rows")

# Get columns from BigBasket_v2
base_columns = df_filtered.columns.tolist()
print(f"\n📋 Base columns in BigBasket_v2: {len(base_columns)}")
print(f"   Columns: {base_columns}")

# Check which columns exist in eggs dataset
eggs_columns = df_eggs.columns.tolist()
print(f"\n📋 Columns in BB_eggs_meat_fish_processed: {len(eggs_columns)}")
print(f"   Columns: {eggs_columns}")

# Find extra columns in eggs dataset
extra_columns = set(eggs_columns) - set(base_columns)
if extra_columns:
    print(f"\n⚠️  Extra columns in eggs dataset (will be dropped): {extra_columns}")
    df_eggs = df_eggs[base_columns].copy()
    print(f"✓ Dropped extra columns")

# Find missing columns
missing_columns = set(base_columns) - set(eggs_columns)
if missing_columns:
    print(f"\n⚠️  Missing columns in eggs dataset: {missing_columns}")
    for col in missing_columns:
        df_eggs[col] = None
        print(f"  Added column '{col}' with None values")

# Reorder columns to match
df_eggs = df_eggs[base_columns]

# Step 5: Verify the eggs data has correct category
print(f"\n✅ Step 5: Verifying 'Eggs, Meat & Fish' category...")
unique_categories = df_eggs['category'].unique()
print(f"   Categories in eggs dataset: {unique_categories}")

if 'Eggs, Meat & Fish' not in unique_categories:
    print("⚠️  WARNING: 'Eggs, Meat & Fish' not found in category column!")
    print(f"   Found categories: {unique_categories}")
    # Check if there's a similar category name
    for cat in unique_categories:
        if 'egg' in cat.lower() or 'meat' in cat.lower() or 'fish' in cat.lower():
            print(f"   Possible match: '{cat}'")

# Step 6: Merge the datasets
print(f"\n🔗 Step 6: Merging datasets...")
df_v3 = pd.concat([df_filtered, df_eggs], ignore_index=True)
print(f"✓ Merged successfully")
print(f"  Total rows: {len(df_v3):,}")
print(f"  = {len(df_filtered):,} (other categories) + {len(df_eggs):,} (Eggs/Meat/Fish)")

# Step 7: Verify final dataset
print(f"\n📊 Step 7: Final category distribution:")
final_counts = df_v3['category'].value_counts()
for cat, count in final_counts.items():
    print(f"  {cat:<40} {count:>6,} rows")

# Verify no nulls in critical columns
print(f"\n🔍 Checking for missing values in critical columns:")
critical_cols = ['sale_price', 'rating_clean', 'category']
for col in critical_cols:
    if col in df_v3.columns:
        n_missing = df_v3[col].isna().sum()
        print(f"  {col:<20} {n_missing:>6} missing ({n_missing/len(df_v3)*100:.1f}%)")

# Step 8: Save the merged dataset
output_file = 'data/processed/BigBasket_v3.csv'
print(f"\n💾 Step 8: Saving to {output_file}...")
df_v3.to_csv(output_file, index=False)
print(f"✓ Saved successfully")

# Summary
print("\n" + "="*80)
print("✅ BIGBASKET_V3.CSV CREATED SUCCESSFULLY!")
print("="*80)
print(f"\n📈 Summary:")
print(f"  Original rows (v2):           {len(df_v2):>8,}")
print(f"  - Eggs, Meat & Fish removed:  {len(df_v2) - len(df_without_eggs):>8,}")
print(f"  - Fruits & Vegetables removed: {len(df_without_eggs) - len(df_filtered):>8,}")
print(f"  + Processed Eggs/Meat/Fish:   {len(df_eggs):>8,}")
print(f"  = Final rows (v3):            {len(df_v3):>8,}")
print(f"\n  Total categories: {len(final_counts)}")
print(f"  Output file: {output_file}")
print("\n🎯 Ready for week1_FIXED.py!")
