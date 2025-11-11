"""
Convert E-commerce Customer Behavior Dataset to RFM Format
===========================================================

This script converts the e-commerce customer behavior CSV into RFM format
compatible with the Transfer Learning Framework CLI.

Your dataset columns:
- ID: Customer ID
- Gender: Male/Female
- Age: Customer age
- City: Customer city
- Membership Type: Gold/Silver/Bronze
- Total Spend: Total amount spent (maps to Monetary)
- Items Purchased: Number of items (maps to Frequency)
- Average Rating: Rating given
- Discount Applied: TRUE/FALSE
- Days Since Last Purchase: Days since last purchase (maps to Recency)
- Satisfaction Level: Satisfied/Neutral/Unsatisfied

RFM required columns:
- customer_id: Unique identifier
- Recency: Days since last purchase (lower is better)
- Frequency: Number of transactions/items (higher is better)
- Monetary: Total amount spent (higher is better)

Author: Transfer Learning Framework Team
Date: November 11, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Fix Windows console encoding for unicode characters
if sys.platform == 'win32':
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

def convert_ecommerce_to_rfm(input_csv_path, output_csv_path=None, 
                             filter_membership=None, filter_city=None,
                             filter_satisfaction=None):
    """
    Convert e-commerce CSV to RFM format
    
    Parameters:
    -----------
    input_csv_path : str
        Path to the e-commerce customer behavior CSV
    output_csv_path : str, optional
        Path to save the RFM CSV. If None, auto-generates name
    filter_membership : str or list, optional
        Filter by membership type (e.g., 'Gold', ['Gold', 'Silver'])
    filter_city : str or list, optional
        Filter by city (e.g., 'New York', ['New York', 'Los Angeles'])
    filter_satisfaction : str or list, optional
        Filter by satisfaction level (e.g., 'Satisfied', ['Satisfied', 'Neutral'])
    
    Returns:
    --------
    rfm_df : DataFrame
        RFM formatted dataframe
    """
    
    print("="*80)
    print("E-COMMERCE TO RFM CONVERTER")
    print("="*80)
    
    # Load data
    print(f"\nLoading: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Strip whitespace from column names (common issue in CSV files)
    df.columns = df.columns.str.strip()
    
    print(f"[SUCCESS] Loaded {len(df)} customers")
    
    # Display original columns
    print(f"\nOriginal columns: {list(df.columns)}")
    
    # Check for data quality issues
    print(f"\nData Quality Check:")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Duplicate IDs: {df['ID'].duplicated().sum()}")
    
    # Clean and convert numeric columns
    print(f"\nCleaning data types...")
    
    # Clean Total Spend: remove any non-numeric characters and convert to float
    if df['Total Spend'].dtype == 'object':
        print(f"   Cleaning 'Total Spend' column (found non-numeric values)")
        df['Total Spend'] = pd.to_numeric(df['Total Spend'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
        cleaned = df['Total Spend'].isna().sum()
        if cleaned > 0:
            print(f"   [WARNING] Converted {cleaned} invalid Total Spend values to NaN")
    
    # Ensure numeric types for other columns
    df['Days Since Last Purchase'] = pd.to_numeric(df['Days Since Last Purchase'], errors='coerce')
    df['Items Purchased'] = pd.to_numeric(df['Items Purchased'], errors='coerce')
    df['Average Rating'] = pd.to_numeric(df['Average Rating'], errors='coerce')
    
    # Apply filters if specified
    original_count = len(df)
    
    if filter_membership:
        if isinstance(filter_membership, str):
            filter_membership = [filter_membership]
        df = df[df['Membership Type'].isin(filter_membership)]
        print(f"   Filtered by membership {filter_membership}: {len(df)} customers")
    
    if filter_city:
        if isinstance(filter_city, str):
            filter_city = [filter_city]
        df = df[df['City'].isin(filter_city)]
        print(f"   Filtered by city {filter_city}: {len(df)} customers")
    
    if filter_satisfaction:
        if isinstance(filter_satisfaction, str):
            filter_satisfaction = [filter_satisfaction]
        df = df[df['Satisfaction Level'].isin(filter_satisfaction)]
        print(f"   Filtered by satisfaction {filter_satisfaction}: {len(df)} customers")
    
    if len(df) < original_count:
        print(f"\n   Total filtered: {original_count} → {len(df)} customers")
    
    # Create RFM dataframe
    print(f"\nConverting to RFM format...")
    
    rfm_df = pd.DataFrame({
        'customer_id': df['ID'].astype(str),
        'Recency': pd.to_numeric(df['Days Since Last Purchase'], errors='coerce'),
        'Frequency': pd.to_numeric(df['Items Purchased'], errors='coerce'),
        'Monetary': pd.to_numeric(df['Total Spend'], errors='coerce')
    })
    
    # Check for missing values in critical RFM columns BEFORE adding metadata
    missing_before = rfm_df[['Recency', 'Frequency', 'Monetary']].isnull().sum()
    if missing_before.any():
        print(f"\n[WARNING] Missing values in RFM columns:")
        for col, count in missing_before[missing_before > 0].items():
            print(f"   {col}: {count} missing")
        print(f"   Dropping {rfm_df[['Recency', 'Frequency', 'Monetary']].isnull().any(axis=1).sum()} rows with missing values...")
        rfm_df = rfm_df.dropna(subset=['Recency', 'Frequency', 'Monetary'])
        # Update df to match the cleaned rfm_df indices
        df = df.loc[rfm_df.index]
    
    # Add optional metadata columns (useful for analysis but not required)
    rfm_df['gender'] = df['Gender'].values
    rfm_df['age'] = df['Age'].values
    rfm_df['city'] = df['City'].values
    rfm_df['membership_type'] = df['Membership Type'].values
    rfm_df['avg_rating'] = df['Average Rating'].values
    rfm_df['discount_applied'] = df['Discount Applied'].values
    rfm_df['satisfaction'] = df['Satisfaction Level'].fillna('Unknown').values
    
    # Display RFM statistics (ensure numeric types)
    print(f"\nRFM Statistics:")
    print(f"   Customers: {len(rfm_df)}")
    print(f"   Recency   - Min: {rfm_df['Recency'].min():.0f}, Max: {rfm_df['Recency'].max():.0f}, Mean: {rfm_df['Recency'].mean():.1f} days")
    print(f"   Frequency - Min: {rfm_df['Frequency'].min():.0f}, Max: {rfm_df['Frequency'].max():.0f}, Mean: {rfm_df['Frequency'].mean():.1f} items")
    print(f"   Monetary  - Min: ${rfm_df['Monetary'].min():.2f}, Max: ${rfm_df['Monetary'].max():.2f}, Mean: ${rfm_df['Monetary'].mean():.2f}")
    
    # Save to CSV
    if output_csv_path is None:
        # Auto-generate filename
        input_path = Path(input_csv_path)
        output_csv_path = input_path.parent / f"{input_path.stem}_RFM.csv"
    
    rfm_df.to_csv(output_csv_path, index=False)
    print(f"\n[SUCCESS] Saved RFM data: {output_csv_path}")
    print(f"   Total customers: {len(rfm_df)}")
    
    return rfm_df


def create_domain_pairs_from_ecommerce(input_csv_path, output_dir='data/ecommerce'):
    """
    Create source and target domain pairs from the e-commerce dataset
    
    This creates several useful domain pairs:
    1. Gold → Silver (high-value to mid-value customers)
    2. Satisfied → Neutral (satisfied to neutral customers)
    3. New York → Los Angeles (geographic transfer)
    4. High Spenders → Low Spenders (spending tier transfer)
    
    Parameters:
    -----------
    input_csv_path : str
        Path to the e-commerce customer behavior CSV
    output_dir : str
        Directory to save the domain pair files
    """
    
    print("\n" + "="*80)
    print("CREATING DOMAIN PAIRS FROM E-COMMERCE DATA")
    print("="*80)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load original data
    df = pd.read_csv(input_csv_path)
    
    # Clean column names and data types
    df.columns = df.columns.str.strip()
    
    # Clean Total Spend: remove any non-numeric characters and convert to float
    if df['Total Spend'].dtype == 'object':
        df['Total Spend'] = pd.to_numeric(df['Total Spend'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
    
    # Ensure numeric types for other columns
    df['Days Since Last Purchase'] = pd.to_numeric(df['Days Since Last Purchase'], errors='coerce')
    df['Items Purchased'] = pd.to_numeric(df['Items Purchased'], errors='coerce')
    
    print(f"\nLoaded {len(df)} customers from {input_csv_path}")
    
    # Domain Pair 1: Gold Members → Silver Members
    print("\n[Pair 1] Gold Members -> Silver Members")
    source1 = convert_ecommerce_to_rfm(
        input_csv_path, 
        f"{output_dir}/pair1_gold_source_RFM.csv",
        filter_membership='Gold'
    )
    target1 = convert_ecommerce_to_rfm(
        input_csv_path,
        f"{output_dir}/pair1_silver_target_RFM.csv",
        filter_membership='Silver'
    )
    print(f"[SUCCESS] Created Pair 1: {len(source1)} Gold -> {len(target1)} Silver members")
    
    # Domain Pair 2: Satisfied → Neutral Customers
    print("\n[Pair 2] Satisfied -> Neutral Customers")
    source2 = convert_ecommerce_to_rfm(
        input_csv_path,
        f"{output_dir}/pair2_satisfied_source_RFM.csv",
        filter_satisfaction='Satisfied'
    )
    target2 = convert_ecommerce_to_rfm(
        input_csv_path,
        f"{output_dir}/pair2_neutral_target_RFM.csv",
        filter_satisfaction='Neutral'
    )
    print(f"[SUCCESS] Created Pair 2: {len(source2)} Satisfied -> {len(target2)} Neutral customers")
    
    # Domain Pair 3: New York → Los Angeles
    print("\n[Pair 3] New York -> Los Angeles (Geographic Transfer)")
    source3 = convert_ecommerce_to_rfm(
        input_csv_path,
        f"{output_dir}/pair3_newyork_source_RFM.csv",
        filter_city='New York'
    )
    target3 = convert_ecommerce_to_rfm(
        input_csv_path,
        f"{output_dir}/pair3_losangeles_target_RFM.csv",
        filter_city='Los Angeles'
    )
    print(f"[SUCCESS] Created Pair 3: {len(source3)} New York -> {len(target3)} Los Angeles customers")
    
    # Domain Pair 4: High Spenders → Low Spenders
    print("\n[Pair 4] High Spenders -> Low Spenders")
    median_spend = df['Total Spend'].median()
    
    # High spenders (top 50%)
    high_spenders = df[df['Total Spend'] >= median_spend]
    high_spenders.to_csv('temp_high_spenders.csv', index=False)
    source4 = convert_ecommerce_to_rfm(
        'temp_high_spenders.csv',
        f"{output_dir}/pair4_highspend_source_RFM.csv"
    )
    
    # Low spenders (bottom 50%)
    low_spenders = df[df['Total Spend'] < median_spend]
    low_spenders.to_csv('temp_low_spenders.csv', index=False)
    target4 = convert_ecommerce_to_rfm(
        'temp_low_spenders.csv',
        f"{output_dir}/pair4_lowspend_target_RFM.csv"
    )
    
    # Clean up temp files
    import os
    os.remove('temp_high_spenders.csv')
    os.remove('temp_low_spenders.csv')
    
    print(f"[SUCCESS] Created Pair 4: {len(source4)} High Spenders (>${median_spend:.2f}) -> {len(target4)} Low Spenders")
    
    print("\n" + "="*80)
    print("[SUCCESS] ALL DOMAIN PAIRS CREATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nYou can now run the CLI on these domain pairs:")
    print(f"\n   Example 1 (Gold → Silver):")
    print(f"   python src/week3/cli.py --mode rfm \\")
    print(f"       --source {output_dir}/pair1_gold_source_RFM.csv \\")
    print(f"       --target {output_dir}/pair1_silver_target_RFM.csv")
    print(f"\n   Example 2 (New York → Los Angeles):")
    print(f"   python src/week3/cli.py --mode rfm \\")
    print(f"       --source {output_dir}/pair3_newyork_source_RFM.csv \\")
    print(f"       --target {output_dir}/pair3_losangeles_target_RFM.csv")
    
    return {
        'pair1': {'source': source1, 'target': target1, 'name': 'Gold → Silver'},
        'pair2': {'source': source2, 'target': target2, 'name': 'Satisfied → Neutral'},
        'pair3': {'source': source3, 'target': target3, 'name': 'New York → Los Angeles'},
        'pair4': {'source': source4, 'target': target4, 'name': 'High Spend → Low Spend'}
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert E-commerce Customer Behavior CSV to RFM format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert entire dataset
  python convert_ecommerce_to_rfm.py input.csv
  
  # Convert with filters
  python convert_ecommerce_to_rfm.py input.csv --membership Gold --city "New York"
  
  # Create multiple domain pairs automatically
  python convert_ecommerce_to_rfm.py input.csv --create-pairs
        """
    )
    
    parser.add_argument('input_csv', help='Path to e-commerce CSV file')
    parser.add_argument('-o', '--output', help='Output RFM CSV path (optional)')
    parser.add_argument('--membership', choices=['Gold', 'Silver', 'Bronze'], 
                       help='Filter by membership type')
    parser.add_argument('--city', help='Filter by city')
    parser.add_argument('--satisfaction', choices=['Satisfied', 'Neutral', 'Unsatisfied'],
                       help='Filter by satisfaction level')
    parser.add_argument('--create-pairs', action='store_true',
                       help='Automatically create 4 domain pairs')
    parser.add_argument('--output-dir', default='data/ecommerce',
                       help='Output directory for domain pairs (default: data/ecommerce)')
    
    args = parser.parse_args()
    
    if args.create_pairs:
        # Create multiple domain pairs
        create_domain_pairs_from_ecommerce(args.input_csv, args.output_dir)
    else:
        # Single conversion
        rfm_df = convert_ecommerce_to_rfm(
            args.input_csv,
            args.output,
            filter_membership=args.membership,
            filter_city=args.city,
            filter_satisfaction=args.satisfaction
        )
        
        print("\n[SUCCESS] Conversion complete!")
        print("\nNext steps:")
        print(f"   1. Create a target domain RFM file (with different filters)")
        print(f"   2. Run the CLI:")
        print(f"      python src/week3/cli.py --mode rfm \\")
        print(f"          --source <source_rfm.csv> \\")
        print(f"          --target <target_rfm.csv>")
