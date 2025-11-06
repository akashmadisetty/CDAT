"""
Process BB_eggs_meat_fish_rated.csv
- Split tuple in rating_clean column
- Extract URL to new column
- Extract rating and fill missing with random values (3.0-3.3)
"""

import pandas as pd
import numpy as np
import ast
import re

print("="*80)
print("PROCESSING BB_eggs_meat_fish_rated.csv")
print("="*80)

# Load the CSV
df = pd.read_csv('BB_eggs_meat_fish_rated.csv')
print(f"\n✓ Loaded {len(df):,} rows")
print(f"Columns: {list(df.columns)}")

# Check the rating_clean column
print(f"\n📊 Sample rating_clean values:")
print(df['rating_clean'].head(10))

# Function to safely parse the tuple string
def parse_rating_tuple(rating_str):
    """
    Parse the rating_clean tuple string
    Returns: (url, rating_value)
    """
    try:
        # Handle if it's already parsed somehow
        if pd.isna(rating_str):
            return (None, None)
        
        # Convert string to actual tuple
        rating_str = str(rating_str).strip()
        
        # Use ast.literal_eval to safely evaluate the tuple
        parsed = ast.literal_eval(rating_str)
        
        if isinstance(parsed, tuple) and len(parsed) >= 2:
            url = parsed[0]
            rating = parsed[1]
            
            # Convert rating to float if it's not None
            if rating is not None and rating != 'None':
                try:
                    rating = float(rating)
                except (ValueError, TypeError):
                    rating = None
            else:
                rating = None
            
            return (url, rating)
        else:
            return (None, None)
            
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"  ⚠️  Error parsing: {rating_str[:50]}... Error: {e}")
        return (None, None)

# Apply parsing
print("\n🔄 Parsing rating_clean tuples...")
parsed_data = df['rating_clean'].apply(parse_rating_tuple)

# Extract URL and rating into separate columns
df['product_url'] = parsed_data.apply(lambda x: x[0] if x[0] is not None else 'n/a')
df['rating_value'] = parsed_data.apply(lambda x: x[1])

print(f"✓ Extracted URLs and ratings")

# Count missing values
n_missing_urls = (df['product_url'] == 'n/a').sum()
n_missing_ratings = df['rating_value'].isna().sum()

print(f"\n📈 Missing values:")
print(f"  URLs: {n_missing_urls} / {len(df)} ({n_missing_urls/len(df)*100:.1f}%)")
print(f"  Ratings: {n_missing_ratings} / {len(df)} ({n_missing_ratings/len(df)*100:.1f}%)")


np.random.seed(42)  # For reproducibility
random_ratings = np.random.uniform(2.8, 3.2, size=n_missing_ratings)

print(f"\n🎲 Generating {n_missing_ratings} random ratings between 2.8 - 3.2...")
df.loc[df['rating_value'].isna(), 'rating_value'] = random_ratings

# Round all ratings to one decimal place
df['rating_value'] = df['rating_value'].round(1)

# Update the rating_clean column with just the numeric rating (rounded)
df['rating_clean'] = df['rating_value']

print(f"✓ Filled all missing ratings and rounded to 1 decimal place")

# Verify no missing values in rating_clean
n_still_missing = df['rating_clean'].isna().sum()
print(f"\n✅ Final check: {n_still_missing} missing values in rating_clean")

# Show statistics
print(f"\n📊 Rating statistics:")
print(f"  Min: {df['rating_clean'].min():.2f}")
print(f"  Max: {df['rating_clean'].max():.2f}")
print(f"  Mean: {df['rating_clean'].mean():.2f}")
print(f"  Median: {df['rating_clean'].median():.2f}")

# Show sample of processed data
print(f"\n📋 Sample of processed data:")
print(df[['product', 'product_url', 'rating_value', 'rating_clean']].head(10))

# Save the processed file
output_file = 'BB_eggs_meat_fish_processed.csv'
df.to_csv(output_file, index=False)
print(f"\n✅ Saved processed data to: {output_file}")

# Show summary of URL types
print(f"\n🔗 URL Summary:")
print(f"  Valid URLs: {(df['product_url'] != 'n/a').sum()}")
print(f"  n/a URLs: {(df['product_url'] == 'n/a').sum()}")

# Show distribution of ratings
print(f"\n⭐ Rating distribution:")
rating_bins = pd.cut(df['rating_clean'], bins=[0, 3.0, 3.5, 4.0, 4.5, 5.0])
print(rating_bins.value_counts().sort_index())

print("\n" + "="*80)
print("✅ PROCESSING COMPLETE!")
print("="*80)
print(f"\nOutput file: {output_file}")
print(f"Total rows: {len(df):,}")
print(f"All rating_clean values filled: {'Yes' if n_still_missing == 0 else 'No'}")
