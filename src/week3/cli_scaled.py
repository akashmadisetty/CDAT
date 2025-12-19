#!/usr/bin/env python3
"""
Transfer Learning Framework - Command Line Interface
Member 4 Deliverable

A multifunctional CLI tool that supports:
1. Built-in domain pairs (from experiment_config.py)
2. Custom RFM CSV files
3. Custom transaction CSV files (auto-generates RFM)
4. External datasets

Usage Examples:
  # Built-in domain pair
  python cli.py --mode builtin --pair 1
  
  # Custom RFM files
  python cli.py --mode rfm --source source_rfm.csv --target target_rfm.csv
  
  # Transaction files (auto-generates RFM)
  python cli.py --mode transactions --source source_trans.csv --target target_trans.csv
  
  # External dataset
  python cli.py --mode external --dataset uk_retail --source UK --target France
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src/week3 to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from experiment_config import DOMAIN_PAIRS, PATHS
from framework import TransferLearningFramework
from decision_engine import DecisionEngine

# ============================================================================
# STYLING & OUTPUT
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text):
    """Print styled header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_builtin_domain_pair(pair_number):
    """Load built-in domain pair from experiment data"""
    print_header(f"Loading Built-in Domain Pair {pair_number}")
    
    if pair_number not in DOMAIN_PAIRS:
        print_error(f"Invalid pair number. Choose from: {list(DOMAIN_PAIRS.keys())}")
        return None, None, None
    
    pair_info = DOMAIN_PAIRS[pair_number]
    pair_info['pair_number'] = pair_number  # Add for model loading
    print_info(f"Pair: {pair_info['name']}")
    print_info(f"Expected Transferability: {pair_info['expected_transferability']}")
    print_info(f"Score: {pair_info['transferability_score']:.4f}")
    
    # Load RFM data
    source_file = f"{PATHS['data_dir']}/domain_pair{pair_number}_source_RFM.csv"
    target_file = f"{PATHS['data_dir']}/domain_pair{pair_number}_target_RFM.csv"
    
    try:
        source_data = pd.read_csv(source_file)
        target_data = pd.read_csv(target_file)
        
        print_success(f"Loaded source data: {len(source_data)} customers")
        print_success(f"Loaded target data: {len(target_data)} customers")
        
        return source_data, target_data, pair_info
    
    except FileNotFoundError as e:
        print_error(f"Data files not found: {e}")
        return None, None, None

def load_rfm_files(source_path, target_path):
    """Load custom RFM CSV files"""
    print_header("Loading Custom RFM Files")
    
    try:
        source_data = pd.read_csv(source_path)
        target_data = pd.read_csv(target_path)
        
        # Validate RFM columns
        required_cols = ['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']
        
        for col in required_cols:
            if col not in source_data.columns:
                print_error(f"Source file missing column: {col}")
                return None, None, None
            if col not in target_data.columns:
                print_error(f"Target file missing column: {col}")
                return None, None, None
        
        print_success(f"Loaded source RFM: {len(source_data)} customers")
        print_success(f"Loaded target RFM: {len(target_data)} customers")
        
        # Create basic pair info
        pair_info = {
            'name': f'Custom: {Path(source_path).stem} → {Path(target_path).stem}',
            'expected_transferability': 'UNKNOWN',
            'transferability_score': None  # Will be calculated
        }
        
        return source_data, target_data, pair_info
    
    except Exception as e:
        print_error(f"Error loading RFM files: {e}")
        return None, None, None

def calculate_rfm_from_transactions(transactions_df, customer_col='customer_id', 
                                    date_col='InvoiceDate', amount_col='amount'):
    """Calculate RFM features from transaction data"""
    print_info("Calculating RFM features from transactions...")
    
    # Parse dates if needed
    if transactions_df[date_col].dtype == 'object':
        transactions_df[date_col] = pd.to_datetime(transactions_df[date_col])
    
    # Reference date (most recent transaction + 1 day)
    reference_date = transactions_df[date_col].max() + pd.Timedelta(days=1)
    
    # Calculate RFM
    rfm = transactions_df.groupby(customer_col).agg({
        date_col: lambda x: (reference_date - x.max()).days,  # Recency_scaled_scaled
        customer_col: 'count',  # Frequency_scaled
        amount_col: 'sum'  # Monetary_scaled
    }).reset_index()
    
    rfm.columns = ['customer_id', 'Recency_scaled_scaled', 'Frequency_scaled', 'Monetary_scaled']
    
    print_success(f"Calculated RFM for {len(rfm)} customers")
    print_info(f"  Recency_scaled_scaled range: [{rfm['Recency_scaled_scaled'].min():.0f}, {rfm['Recency_scaled_scaled'].max():.0f}] days")
    print_info(f"  Frequency_scaled range: [{rfm['Frequency_scaled'].min():.0f}, {rfm['Frequency_scaled'].max():.0f}] purchases")
    print_info(f"  Monetary_scaled range: [₹{rfm['Monetary_scaled'].min():.2f}, ₹{rfm['Monetary_scaled'].max():.2f}]")
    
    return rfm

def load_transaction_files(source_path, target_path):
    """Load transaction files and calculate RFM"""
    print_header("Loading Transaction Files")
    
    try:
        source_trans = pd.read_csv(source_path)
        target_trans = pd.read_csv(target_path)
        
        print_success(f"Loaded source transactions: {len(source_trans)} records")
        print_success(f"Loaded target transactions: {len(target_trans)} records")
        
        # Auto-detect column names
        # Common patterns: customer_id, CustomerID, user_id, etc.
        customer_cols = ['customer_id', 'CustomerID', 'user_id', 'Customer ID']
        date_cols = ['InvoiceDate', 'date', 'transaction_date', 'Date', 'order_date']
        amount_cols = ['amount', 'total', 'sale_price', 'Amount', 'Total']
        
        # Find matching columns
        customer_col = next((c for c in customer_cols if c in source_trans.columns), None)
        date_col = next((c for c in date_cols if c in source_trans.columns), None)
        amount_col = next((c for c in amount_cols if c in source_trans.columns), None)
        
        if not all([customer_col, date_col, amount_col]):
            print_error("Could not auto-detect transaction columns.")
            print_info("Required columns: customer_id, date, amount (or similar)")
            print_info(f"Available columns: {list(source_trans.columns)}")
            return None, None, None
        
        print_info(f"Using columns: customer={customer_col}, date={date_col}, amount={amount_col}")
        
        # Calculate RFM
        source_rfm = calculate_rfm_from_transactions(source_trans, customer_col, date_col, amount_col)
        target_rfm = calculate_rfm_from_transactions(target_trans, customer_col, date_col, amount_col)
        
        pair_info = {
            'name': f'Custom: {Path(source_path).stem} → {Path(target_path).stem}',
            'expected_transferability': 'UNKNOWN',
            'transferability_score': None
        }
        
        return source_rfm, target_rfm, pair_info
    
    except Exception as e:
        print_error(f"Error loading transaction files: {e}")
        return None, None, None

# ============================================================================
# ANALYSIS & RECOMMENDATION
# ============================================================================

def analyze_transferability(source_data, target_data, pair_info, use_learned_weights=True, with_confidence=False):
    """Analyze transferability and generate recommendations"""
    print_header("Transfer Learning Analysis")
    
    # Load source model if available
    source_model = None
    if 'pair_number' in pair_info:
        model_path = f"{PATHS['models_dir']}/domain_pair{pair_info['pair_number']}_rfm_kmeans_model.pkl"
        if os.path.exists(model_path):
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                source_model = model_data.get('model')
                print_success(f"Loaded pre-trained source model")
    
    # Create framework instance
    framework = TransferLearningFramework(
        source_model=source_model,
        source_data=source_data,
        target_data=target_data,
        use_learned_weights=use_learned_weights
    )
    
    # Calculate transferability
    if use_learned_weights:
        print_info("Calculating transferability metrics (using learned weights)...")
    else:
        print_info("Calculating transferability metrics (using default weights)...")
    framework.calculate_transferability()
    
    # Always use the freshly calculated score
    transferability_score = framework.composite_score
    print_success(f"Transferability Score: {transferability_score:.4f}")
    
    # Calculate confidence interval if requested
    if with_confidence:
        print_info("Calculating 95% confidence interval (this may take a minute)...")
        ci = framework.calculate_confidence_interval(n_bootstrap=500, verbose=True)
        
        print(f"\n{Colors.BOLD}Transferability Score with 95% Confidence Interval:{Colors.END}")
        print(f"  Score: {ci['score']:.4f}")
        print(f"  95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        print(f"  Standard Error: {ci['std_error']:.4f}")
        print(f"  CI Width: {ci['ci_upper'] - ci['ci_lower']:.4f}")
    
    # Get recommendation
    recommendation = framework.recommend_strategy(verbose=False)
    
    # Display results
    print_header("Recommendation Summary")
    
    print(f"{Colors.BOLD}Strategy:{Colors.END} {Colors.GREEN}{recommendation.strategy.value.replace('_', ' ').title()}{Colors.END}")
    print(f"{Colors.BOLD}Transferability Level:{Colors.END} {recommendation.transferability_level.value}")
    print(f"{Colors.BOLD}Confidence:{Colors.END} {recommendation.confidence:.1f}%")
    print(f"{Colors.BOLD}Target Data Required:{Colors.END} {recommendation.target_data_percentage}%")
    
    print(f"\n{Colors.BOLD}Reasoning:{Colors.END}")
    print(f"  {recommendation.reasoning}")
    
    if recommendation.risks:
        print(f"\n{Colors.BOLD}{Colors.YELLOW}Potential Risks:{Colors.END}")
        for risk in recommendation.risks:
            print(f"  • {risk}")
    
    print(f"\n{Colors.BOLD}Expected Performance:{Colors.END}")
    print(f"  {recommendation.expected_performance}")
    
    return recommendation

def display_comparison_table(source_data, target_data):
    """Display comparison of source vs target domains"""
    print_header("Domain Comparison")
    
    # Calculate statistics
    stats = pd.DataFrame({
        'Metric': ['Customers', 'Avg Recency_scaled (days)', 'Avg Frequency_scaled', 'Avg Monetary_scaled (₹)', 
                   'Median Recency_scaled', 'Median Frequency_scaled', 'Median Monetary_scaled'],
        'Source Domain': [
            len(source_data),
            source_data['Recency_scaled'].mean(),
            source_data['Frequency_scaled'].mean(),
            source_data['Monetary_scaled'].mean(),
            source_data['Recency_scaled'].median(),
            source_data['Frequency_scaled'].median(),
            source_data['Monetary_scaled'].median()
        ],
        'Target Domain': [
            len(target_data),
            target_data['Recency_scaled'].mean(),
            target_data['Frequency_scaled'].mean(),
            target_data['Monetary_scaled'].mean(),
            target_data['Recency_scaled'].median(),
            target_data['Frequency_scaled'].median(),
            target_data['Monetary_scaled'].median()
        ]
    })
    
    # Format numbers
    stats['Source Domain'] = stats['Source Domain'].apply(lambda x: f"{x:.2f}" if x < 1000 else f"{x:.0f}")
    stats['Target Domain'] = stats['Target Domain'].apply(lambda x: f"{x:.2f}" if x < 1000 else f"{x:.0f}")
    
    print(stats.to_string(index=False))

# ============================================================================
# MAIN CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Transfer Learning Framework CLI - Analyze domain transferability',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use built-in domain pair
  python cli.py --mode builtin --pair 1
  
  # Custom RFM files
  python cli.py --mode rfm --source my_source_rfm.csv --target my_target_rfm.csv
  
  # Transaction files (auto-calculates RFM)
  python cli.py --mode transactions --source src_trans.csv --target tgt_trans.csv
  
  # Show all available built-in pairs
  python cli.py --list-pairs
        '''
    )
    
    parser.add_argument('--mode', choices=['builtin', 'rfm', 'transactions'],
                       help='Input mode: builtin domain pair, RFM files, or transaction files')
    parser.add_argument('--pair', type=int, 
                       help='Built-in domain pair number (1-7)')
    parser.add_argument('--source', type=str,
                       help='Path to source RFM/transaction CSV file')
    parser.add_argument('--target', type=str,
                       help='Path to target RFM/transaction CSV file')
    parser.add_argument('--list-pairs', action='store_true',
                       help='List all available built-in domain pairs')
    parser.add_argument('--no-comparison', action='store_true',
                       help='Skip domain comparison table')
    parser.add_argument('--save-report', type=str,
                       help='Save detailed report to file')
    parser.add_argument('--use-learned-weights', action='store_true', default=True,
                       help='Use learned composite weights from calibration (default: True)')
    parser.add_argument('--use-default-weights', dest='use_learned_weights', action='store_false',
                       help='Use default research-backed weights instead of learned weights')
    parser.add_argument('--with-confidence', action='store_true',
                       help='Calculate 95%% confidence interval using bootstrap (slower, ~1 min)')
    
    args = parser.parse_args()
    
    # Banner
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                   TRANSFER LEARNING FRAMEWORK CLI                             ║")
    print("║                   Customer Segmentation Transfer Analysis                     ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")
    print(Colors.END)
    
    # List pairs mode
    if args.list_pairs:
        print_header("Available Built-in Domain Pairs")
        for pair_num, info in sorted(DOMAIN_PAIRS.items()):
            print(f"\n{Colors.BOLD}Pair {pair_num}:{Colors.END} {info['name']}")
            print(f"  Category: {info['expected_transferability']}")
            print(f"  Transferability Score: {info['transferability_score']:.4f}")
        print()
        return
    
    # Validate arguments
    if not args.mode:
        print_error("Please specify --mode (builtin, rfm, or transactions)")
        print_info("Use --help for usage examples")
        return
    
    # Load data based on mode
    source_data, target_data, pair_info = None, None, None
    
    if args.mode == 'builtin':
        if not args.pair:
            print_error("--pair number required for builtin mode")
            return
        source_data, target_data, pair_info = load_builtin_domain_pair(args.pair)
    
    elif args.mode == 'rfm':
        if not args.source or not args.target:
            print_error("--source and --target required for RFM mode")
            return
        source_data, target_data, pair_info = load_rfm_files(args.source, args.target)
    
    elif args.mode == 'transactions':
        if not args.source or not args.target:
            print_error("--source and --target required for transactions mode")
            return
        source_data, target_data, pair_info = load_transaction_files(args.source, args.target)
    
    # Check if data loaded successfully
    if source_data is None or target_data is None:
        print_error("Failed to load data. Exiting.")
        return
    
    # Display comparison
    if not args.no_comparison:
        display_comparison_table(source_data, target_data)
    
    # Analyze transferability
    recommendation = analyze_transferability(source_data, target_data, pair_info, 
                                            use_learned_weights=args.use_learned_weights,
                                            with_confidence=args.with_confidence)
    
    # Save report if requested
    if args.save_report:
        with open(args.save_report, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TRANSFER LEARNING FRAMEWORK - ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Domain Pair: {pair_info['name']}\n")
            f.write(f"Transferability Score: {pair_info['transferability_score']:.4f}\n\n")
            f.write(str(recommendation))  # Use __str__() method
        
        print_success(f"\nReport saved to: {args.save_report}")
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}Analysis complete!{Colors.END}\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
