import pandas as pd
import numpy as np
from pathlib import Path

# Load the dataset
csv_path = r"d:\Akash\B.Tech\5th Sem\ADA\CDAT\data\processed\BigBasket.csv"
df = pd.read_csv(csv_path)

print("Dataset loaded. Shape:", df.shape)

# Operation 1: Check for empty 'product' column
empty_products = df[df['product'].isna() | (df['product'] == '')]
print(f"\nRows with empty 'product': {len(empty_products)}")

# Operation 2: Fill missing descriptions with 'N/A'
initial_missing_desc = df['description'].isna().sum()
df['description'] = df['description'].fillna('N/A')
df['description'] = df['description'].replace('', 'N/A')
print(f"Rows with missing description filled: {initial_missing_desc}")

# Operation 3: For 'Fruits & Vegetables' and 'Eggs, Meat & Fish' categories, set rating to 3
fruits_veg_mask = df['category'] == 'Fruits & Vegetables'
eggs_meat_mask = df['category'] == 'Eggs, Meat & Fish'

fruits_veg_count = fruits_veg_mask.sum()
eggs_meat_count = eggs_meat_mask.sum()

df.loc[fruits_veg_mask, 'rating'] = 3.0
df.loc[eggs_meat_mask, 'rating'] = 3.0

print(f"Ratings set to 3 for 'Fruits & Vegetables': {fruits_veg_count} rows")
print(f"Ratings set to 3 for 'Eggs, Meat & Fish': {eggs_meat_count} rows")

# Operation 4: For each category and sub_category, fill missing ratings with median
print("\nProcessing missing ratings by category and sub_category...")
missing_rating_filled = 0

for category in df['category'].unique():
    for sub_category in df[df['category'] == category]['sub_category'].unique():
        mask = (df['category'] == category) & (df['sub_category'] == sub_category)
        subset = df[mask].copy()
        
        # Check for missing ratings
        missing_mask = subset['rating'].isna()
        if missing_mask.any():
            # Calculate median of existing ratings
            existing_ratings = subset[~missing_mask]['rating']
            if len(existing_ratings) > 0:
                median_rating = existing_ratings.median()
                df.loc[mask & df['rating'].isna(), 'rating'] = median_rating
                count_filled = missing_mask.sum()
                missing_rating_filled += count_filled
                print(f"  {category} -> {sub_category}: Filled {count_filled} missing with median {median_rating}")

print(f"\nTotal missing ratings filled: {missing_rating_filled}")

# Verification
print("\n=== VERIFICATION ===")
print(f"Missing ratings after processing: {df['rating'].isna().sum()}")
print(f"Missing descriptions after processing: {df['description'].isna().sum()}")
print(f"Empty products: {(df['product'].isna() | (df['product'] == '')).sum()}")

# Save the updated dataset back to the same location
df.to_csv(csv_path, index=False)
print(f"\nDataset updated and saved to: {csv_path}")
print(f"Final dataset shape: {df.shape}")
