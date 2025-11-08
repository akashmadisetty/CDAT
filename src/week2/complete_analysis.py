"""
Complete RFM Analysis Pipeline Runner
Runs training, evaluation, and report generation in one go
"""

import os
import sys
from datetime import datetime


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80)


def print_step(step_num, total_steps, description):
    """Print step information"""
    print(f"\n{'─'*80}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'─'*80}")


def check_data_files():
    """Check if RFM data files exist"""
    print_step(0, 4, "Pre-flight Check: Verifying Data Files")
    
    required_files = [
        'domain_pair1_source_RFM.csv',
        'domain_pair2_source_RFM.csv',
        'domain_pair3_source_RFM.csv',
        'domain_pair4_source_RFM.csv',
        'domain_pair5_source_RFM.csv',
        'domain_pair6_source_RFM.csv',
        'domain_pair7_source_RFM.csv',
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✓ Found: {file}")
        else:
            print(f"   ✗ Missing: {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ ERROR: {len(missing_files)} data file(s) missing!")
        print("   Please ensure all RFM CSV files are in the current directory.")
        print("\n   Missing files:")
        for f in missing_files:
            print(f"      • {f}")
        return False
    
    print("\n✅ All data files found!")
    return True


def run_training():
    """Run the training script"""
    print_step(1, 4, "Training RFM Clustering Models")
    
    try:
        # Import and run training
        import improved_train_all_domains
        improved_train_all_domains.main()
        print("\n✅ Training completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_report_generation():
    """Run the report generator"""
    print_step(2, 4, "Generating Domain-Wise Analysis Reports")
    
    try:
        # Import and run report generator
        import generate_domain_reports
        generate_domain_reports.main()
        print("\n✅ Report generation completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Report generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def create_summary_index():
    """Create an index file summarizing all outputs"""
    print_step(3, 4, "Creating Summary Index")
    
    index = []
    index.append("="*80)
    index.append("RFM CUSTOMER SEGMENTATION - ANALYSIS SUMMARY INDEX")
    index.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    index.append("="*80)
    
    index.append("\n📁 DIRECTORY STRUCTURE:")
    index.append("   models/       - Trained K-Means clustering models (.pkl)")
    index.append("   results/      - Performance metrics, segment profiles, customer segments")
    index.append("   plots/        - Visualizations (elbow curves, 3D plots, profiles)")
    index.append("   reports/      - Comprehensive analysis reports and dashboards")
    
    index.append("\n📊 KEY OUTPUT FILES:")
    
    index.append("\n1. CROSS-DOMAIN ANALYSIS:")
    index.append("   • reports/cross_domain_summary_report.txt")
    index.append("     → Overall comparison of all domains")
    index.append("     → Transfer learning readiness assessment")
    index.append("     → Best/worst performers")
    index.append("\n   • reports/cross_domain_comparative_summary.png")
    index.append("     → Visual dashboard comparing all domains")
    
    index.append("\n2. INDIVIDUAL DOMAIN REPORTS (7 domains):")
    index.append("   • reports/domain_pair[1-7]_analysis_report.txt")
    index.append("     → Detailed analysis for each domain")
    index.append("     → Segment profiles and characteristics")
    index.append("     → Marketing recommendations")
    index.append("     → Action items")
    index.append("\n   • reports/domain_pair[1-7]_dashboard.png")
    index.append("     → Visual dashboard for each domain")
    
    index.append("\n3. TECHNICAL RESULTS:")
    index.append("   • results/improved_baseline_performance.csv")
    index.append("     → All clustering metrics")
    index.append("     → Quality scores")
    index.append("     → Segment statistics")
    index.append("\n   • results/domain_pair[1-7]_segment_profiles.csv")
    index.append("     → Segment characteristics (RFM means, value scores)")
    index.append("\n   • results/domain_pair[1-7]_customer_segments.csv")
    index.append("     → Customer-level segment assignments")
    
    index.append("\n4. VISUALIZATIONS:")
    index.append("   • plots/domain_pair[1-7]_elbow_curve.png")
    index.append("     → Optimal k selection visualization")
    index.append("   • plots/domain_pair[1-7]_rfm_3d.png")
    index.append("     → 3D scatter plot of segments")
    index.append("   • plots/domain_pair[1-7]_segment_profiles.png")
    index.append("     → RFM profile comparison")
    index.append("   • plots/domain_pair[1-7]_distribution.png")
    index.append("     → Customer distribution by segment")
    
    index.append("\n📖 RECOMMENDED READING ORDER:")
    index.append("   1. START HERE: reports/cross_domain_summary_report.txt")
    index.append("   2. View: reports/cross_domain_comparative_summary.png")
    index.append("   3. Read individual domain reports based on your domain of interest")
    index.append("   4. Use domain dashboards for presentations")
    
    index.append("\n🎯 QUICK WINS - WHAT TO DO NEXT:")
    index.append("   1. Identify your best-performing domain (highest Silhouette Score)")
    index.append("   2. Read that domain's analysis report")
    index.append("   3. Review segment profiles and value scores")
    index.append("   4. Design pilot marketing campaign for top 2 segments")
    index.append("   5. Monitor campaign performance for 2-4 weeks")
    index.append("   6. Expand to other segments based on results")
    
    index.append("\n❓ INTERPRETING THE RESULTS:")
    
    index.append("\nQ: What is a 'good' Silhouette Score?")
    index.append("A: • 0.5+  = Excellent (highly confident in segments)")
    index.append("   • 0.35+ = Good (reliable segmentation)")
    index.append("   • 0.25+ = Acceptable (usable with caution)")
    index.append("   • <0.25 = Poor (consider alternatives)")
    
    index.append("\nQ: Which segments should I prioritize?")
    index.append("A: Focus on segments with:")
    index.append("   1. High Value Scores (70+/100)")
    index.append("   2. Reasonable size (not too small)")
    index.append("   3. Clear actionable characteristics")
    index.append("   Priority: Champions > At Risk > Loyal > Promising")
    
    index.append("\nQ: Can I use this model for other product categories?")
    index.append("A: Check the 'Transfer Learning Assessment' section in each report:")
    index.append("   • 'No Finetune' = Yes, use directly")
    index.append("   • 'Partial' = Yes, but fine-tune with some target data")
    index.append("   • 'New Model' = No, train fresh model")
    
    index.append("\n⚠️  IMPORTANT NOTES:")
    index.append("   • RFM captures only transactional behavior")
    index.append("   • Consider adding demographic/preference data for richer segments")
    index.append("   • Re-train models quarterly as customer behavior evolves")
    index.append("   • Monitor segment stability - customers should move gradually")
    index.append("   • Validate segments with actual campaign performance")
    
    index.append("\n📧 SHARING WITH STAKEHOLDERS:")
    index.append("   For executives:")
    index.append("     → Share cross_domain_summary_report.txt")
    index.append("     → Show cross_domain_comparative_summary.png")
    index.append("\n   For marketing team:")
    index.append("     → Share relevant domain analysis reports")
    index.append("     → Use domain dashboards in presentations")
    index.append("     → Provide segment profile CSVs for campaign planning")
    index.append("\n   For data science team:")
    index.append("     → Share improved_baseline_performance.csv")
    index.append("     → Provide model .pkl files for deployment")
    
    index.append("\n" + "="*80)
    index.append("END OF INDEX")
    index.append("="*80)
    
    # Save index
    with open('ANALYSIS_SUMMARY_INDEX.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(index))
    
    print("\n✅ Created ANALYSIS_SUMMARY_INDEX.txt")
    return True


def print_final_summary():
    """Print final summary and next steps"""
    print_step(4, 4, "Analysis Pipeline Complete!")
    
    print("\n" + "🎉"*40)
    print("SUCCESS! RFM ANALYSIS PIPELINE COMPLETED")
    print("🎉"*40)
    
    print("\n📁 All outputs have been generated:")
    print("   ✓ 7 trained clustering models")
    print("   ✓ 7 domain-specific analysis reports")
    print("   ✓ 7 domain-specific dashboards")
    print("   ✓ 1 cross-domain comparative summary")
    print("   ✓ 28 visualizations (4 per domain)")
    print("   ✓ Performance metrics and segment profiles")
    print("   ✓ Customer-level segment assignments")
    
    print("\n🎯 START HERE:")
    print("   1. Open: ANALYSIS_SUMMARY_INDEX.txt")
    print("      (Complete guide to all outputs)")
    print("\n   2. Read: reports/cross_domain_summary_report.txt")
    print("      (Executive overview)")
    print("\n   3. View: reports/cross_domain_comparative_summary.png")
    print("      (Visual comparison)")
    
    print("\n💡 QUICK ACTION ITEMS:")
    print("   → Schedule meeting with marketing team")
    print("   → Review segment profiles for your domain")
    print("   → Design pilot campaign for top segment")
    print("   → Set campaign budget allocation")
    
    print("\n✅ Analysis pipeline completed successfully!")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """
    Main runner function
    """
    print("\n" + "🚀"*40)
    print("RFM CUSTOMER SEGMENTATION - COMPLETE ANALYSIS PIPELINE")
    print("🚀"*40)
    
    print(f"\nStarting analysis at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 0: Check data files
    if not check_data_files():
        print("\n❌ Pipeline aborted: Missing data files")
        sys.exit(1)
    
    # Step 1: Training
    if not run_training():
        print("\n❌ Pipeline aborted: Training failed")
        sys.exit(1)
    
    # Step 2: Report generation
    if not run_report_generation():
        print("\n❌ Pipeline aborted: Report generation failed")
        sys.exit(1)
    
    # Step 3: Create summary index
    if not create_summary_index():
        print("\n⚠️  Warning: Summary index creation failed")
    
    # Step 4: Final summary
    print_final_summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)