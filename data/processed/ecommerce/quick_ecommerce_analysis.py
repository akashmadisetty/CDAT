"""
Quick Runner: E-commerce Dataset Analysis
==========================================

One-command script to:
1. Convert e-commerce CSV to RFM format
2. Create 4 domain pairs
3. Run transfer learning analysis on all pairs
4. Generate summary report

Usage:
    python quick_ecommerce_analysis.py "path/to/E-commerce Customer Behavior - Sheet1.csv"

Author: Transfer Learning Framework Team
Date: November 11, 2025
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd

# Fix Windows console encoding for unicode characters
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*80}")
    print(f"[RUNNING] {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"[SUCCESS] {description}")
        return True
    else:
        print(result.stderr)
        print(f"[FAILED] {description}")
        return False


def main(ecommerce_csv_path):
    """Run complete analysis pipeline"""
    
    print("="*80)
    print("E-COMMERCE TRANSFER LEARNING ANALYSIS")
    print("Complete Pipeline Execution")
    print("="*80)
    
    ecommerce_csv = Path(ecommerce_csv_path)
    if not ecommerce_csv.exists():
        print(f"[ERROR] File not found: {ecommerce_csv_path}")
        sys.exit(1)
    
    # Create output directories
    output_dir = Path("data/ecommerce")
    results_dir = Path("results/ecommerce")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nInput file: {ecommerce_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Results directory: {results_dir}")
    
    # Step 1: Convert to RFM and create domain pairs
    # If the provided file is already an RFM CSV (filename contains '_RFM'),
    # skip conversion/pair-creation and assume pair files already exist in data/ecommerce.
    rfm_mode = False
    if ecommerce_csv.name.endswith('_RFM.csv') or '_RFM' in ecommerce_csv.stem:
        rfm_mode = True
        print(f"[INFO] Detected RFM input file. Skipping conversion and pair creation.")
        print(f"[INFO] Using existing domain pair files in: {output_dir}")
    else:
        success = run_command(
            f'python convert_ecommerce_to_rfm.py "{ecommerce_csv}" --create-pairs --output-dir {output_dir}',
            "Step 1: Converting to RFM and creating domain pairs"
        )

        if not success:
            print("\n[ERROR] Pipeline failed at Step 1")
            sys.exit(1)
    
    # Step 2: Run analysis on all domain pairs
    # Use CLI's --save-report to produce clean UTF-8 reports (no ANSI escapes)
    domain_pairs = [
        {
            'name': 'Pair 1: Gold → Silver Members',
            'source': f'{output_dir}/pair1_gold_source_RFM.csv',
            'target': f'{output_dir}/pair1_silver_target_RFM.csv',
            'report': f'{results_dir}/pair1_clean_report.txt'
        },
        {
            'name': 'Pair 2: Satisfied → Neutral Customers',
            'source': f'{output_dir}/pair2_satisfied_source_RFM.csv',
            'target': f'{output_dir}/pair2_neutral_target_RFM.csv',
            'report': f'{results_dir}/pair2_clean_report.txt'
        },
        {
            'name': 'Pair 3: New York → Los Angeles',
            'source': f'{output_dir}/pair3_newyork_source_RFM.csv',
            'target': f'{output_dir}/pair3_losangeles_target_RFM.csv',
            'report': f'{results_dir}/pair3_clean_report.txt'
        },
        {
            'name': 'Pair 4: High Spend → Low Spend',
            'source': f'{output_dir}/pair4_highspend_source_RFM.csv',
            'target': f'{output_dir}/pair4_lowspend_target_RFM.csv',
            'report': f'{results_dir}/pair4_clean_report.txt'
        }
    ]
    
    results = []
    
    for i, pair in enumerate(domain_pairs, 1):
        print(f"\n{'='*80}")
        print(f"Step 2.{i}: Analyzing {pair['name']}")
        print(f"{'='*80}")
        
        # Let CLI create a clean report (UTF-8, no ANSI) using --save-report
        cmd = f'python src/week3/cli.py --mode rfm --source {pair["source"]} --target {pair["target"]} --save-report {pair["report"]}'

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace')

        # Print CLI stdout/stderr for visibility (do not save raw stdout)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Check if report file was created successfully
        if Path(pair['report']).exists():
            print(f"[SUCCESS] Analysis complete: {pair['report']}")

            # Extract metrics from the saved clean report
            try:
                with open(pair['report'], 'r', encoding='utf-8') as f:
                    content = f.read()

                    score = None
                    level = None
                    strategy = None

                    for line in content.split('\n'):
                        if 'COMPOSITE TRANSFERABILITY SCORE' in line or 'Composite Score' in line:
                            parts = line.split(':')
                            if len(parts) > 1:
                                try:
                                    score = float(parts[1].strip())
                                except:
                                    pass
                        elif '✓ Transferability Score:' in line and score is None:
                            try:
                                score = float(line.split(':')[1].strip())
                            except:
                                pass
                        elif 'Transferability Level:' in line:
                            level = line.split(':')[1].strip()
                        elif 'Strategy:' in line and 'Expected' not in line:
                            strategy = line.split(':')[1].strip()

                    if score is not None:
                        results.append({
                            'Pair': pair['name'],
                            'Score': f"{score:.4f}",
                            'Level': level or 'N/A',
                            'Strategy': strategy or 'N/A'
                        })
            except Exception as e:
                print(f"[WARNING] Could not extract metrics from {pair['report']}: {e}")
        else:
            print(f"[ERROR] Analysis failed: {pair['name']}")
    
    # Step 3: Generate summary report
    print(f"\n{'='*80}")
    print("Step 3: Generating Summary Report")
    print(f"{'='*80}")
    
    # Debug: show how many result entries we collected
    print(f"\n[DEBUG] Collected result entries: {len(results)}")

    if results:
        summary_df = pd.DataFrame(results)
        summary_df = summary_df.sort_values('Score', ascending=False)
        
        summary_path = f'{results_dir}/SUMMARY_REPORT.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("E-COMMERCE TRANSFER LEARNING ANALYSIS - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dataset: {ecommerce_csv.name}\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Pairs Analyzed: {len(results)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("TRANSFERABILITY RANKING (Best to Worst)\n")
            f.write("="*80 + "\n\n")
            
            for idx, row in summary_df.iterrows():
                f.write(f"{row['Pair']}\n")
                f.write(f"  Score: {row['Score']}\n")
                f.write(f"  Level: {row['Level']}\n")
                f.write(f"  Strategy: {row['Strategy']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("KEY INSIGHTS\n")
            f.write("="*80 + "\n\n")
            
            high_count = sum(1 for r in results if 'HIGH' in r['Level'])
            moderate_count = sum(1 for r in results if 'MODERATE' in r['Level'])
            low_count = sum(1 for r in results if 'LOW' in r['Level'])
            
            f.write(f"Distribution:\n")
            f.write(f"  HIGH Transferability: {high_count} pairs\n")
            f.write(f"  MODERATE Transferability: {moderate_count} pairs\n")
            f.write(f"  LOW Transferability: {low_count} pairs\n\n")
            
            best_pair = summary_df.iloc[0]
            f.write(f"Best Transfer Candidate:\n")
            f.write(f"  {best_pair['Pair']}\n")
            f.write(f"  Score: {best_pair['Score']} ({best_pair['Level']})\n\n")
            
            worst_pair = summary_df.iloc[-1]
            f.write(f"Worst Transfer Candidate:\n")
            f.write(f"  {worst_pair['Pair']}\n")
            f.write(f"  Score: {worst_pair['Score']} ({worst_pair['Level']})\n\n")
            
            f.write("="*80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Individual analysis reports saved in: {results_dir}/\n")
            for pair in domain_pairs:
                # Write the report filenames (we now use CLI --save-report outputs)
                f.write(f"  - {Path(pair['report']).name}\n")
        
        print(f"\n[SUCCESS] Summary report saved: {summary_path}")
        
        # Print summary to console
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE - SUMMARY")
        print(f"{'='*80}\n")
        print(summary_df.to_string(index=False))
        
        print(f"\n{'='*80}")
        print("OUTPUT FILES")
        print(f"{'='*80}")
        print(f"  RFM Data: {output_dir}/")
        print(f"  Analysis Results: {results_dir}/")
        print(f"  Summary Report: {summary_path}")
        print(f"\n[SUCCESS] All done! Check the results directory for detailed analysis.")
    else:
        print("[ERROR] No results to summarize - all analyses failed")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_ecommerce_analysis.py <path_to_ecommerce_csv>")
        print("\nExample:")
        print('  python quick_ecommerce_analysis.py "E-commerce Customer Behavior - Sheet1.csv"')
        sys.exit(1)
    
    ecommerce_csv_path = sys.argv[1]
    main(ecommerce_csv_path)
