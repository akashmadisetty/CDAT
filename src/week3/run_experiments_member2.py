"""
Run Transfer Learning Experiments for Pairs 3 & 4
Member 2 - Week 3 Deliverable
"""

import pandas as pd
from transfer_experiments import TransferLearningExperiment, calculate_source_baseline
import os

# ========================================
# ONLY CHANGE THIS SECTION:
# ========================================
EXPERIMENTS = {
    3: {
        'name': 'Premium → Budget Segment',
        'expected_transfer': 'LOW',
        'week1_score': 0.715,
        'source_rfm': '../week2/domain_pair3_source_RFM.csv',
        'target_rfm': '../week2/domain_pair3_target_RFM.csv',
        'source_model': '../week2/models/domain_pair3_rfm_kmeans_model.pkl'
    },
    4: {
        'name': 'Popular → Niche Brands',
        'expected_transfer': 'LOW-MODERATE',
        'week1_score': 0.874,
        'source_rfm': '../week2/domain_pair4_source_RFM.csv',
        'target_rfm': '../week2/domain_pair4_target_RFM.csv',
        'source_model': '../week2/models/domain_pair4_rfm_kmeans_model.pkl'
    }
}
# ========================================

def run_experiment(pair_id, config):
    """Run all tests for a single domain pair"""
    print("\n" + "="*80)
    print(f"EXPERIMENT {pair_id}: {config['name']}")
    print(f"Expected Transferability: {config['expected_transfer']} (Week 1 Score: {config['week1_score']:.3f})")
    print("="*80)
    
    # Calculate source baseline
    source_metrics = calculate_source_baseline(config['source_rfm'], config['source_model'])
    
    # Run experiment
    experiment = TransferLearningExperiment(
        pair_id=pair_id,
        source_rfm_file=config['source_rfm'],
        target_rfm_file=config['target_rfm'],
        source_model_file=config['source_model']
    )
    
    results_df = experiment.run_all_tests()
    
    # Add metadata to results
    results_df['pair_name'] = config['name']
    results_df['expected_transfer'] = config['expected_transfer']
    results_df['week1_score'] = config['week1_score']
    results_df['source_silhouette'] = source_metrics['silhouette']
    
    # Calculate transfer quality percentage
    scratch_silhouette = results_df[results_df['strategy'] == 'Train from scratch']['silhouette'].values[0]
    results_df['transfer_quality_pct'] = (results_df['silhouette'] / scratch_silhouette) * 100
    
    # Save with all metadata
    output_file = f'results/experiment{pair_id}_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved results with metadata: {output_file}")
    
    return results_df


def main():
    """Main execution pipeline"""
    print("\n" + "🚀"*40)
    print("TRANSFER LEARNING EXPERIMENTS - MEMBER 2 WEEK 3")
    print("Pairs 3 & 4: LOW and LOW-MODERATE Transferability")
    print("🚀"*40)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run Experiment 3
    print("\n" + "🔴"*40)
    print("EXPERIMENT 3: LOW TRANSFERABILITY (Premium → Budget)")
    print("🔴"*40)
    results3 = run_experiment(3, EXPERIMENTS[3])
    
    # Run Experiment 4
    print("\n" + "🟡"*40)
    print("EXPERIMENT 4: LOW-MODERATE TRANSFERABILITY (Popular → Niche)")
    print("🟡"*40)
    results4 = run_experiment(4, EXPERIMENTS[4])
    
    # Combined analysis
    combined = pd.concat([results3, results4], ignore_index=True)
    combined.to_csv('results/experiments_3_4_combined.csv', index=False)
    
    # Summary
    print("\n" + "="*80)
    print("🎉 EXPERIMENTS COMPLETE!")
    print("="*80)
    
    print("\n📊 QUICK SUMMARY:")
    print("\nExperiment 3 (LOW Transferability):")
    transfer_asis_3 = results3[results3['strategy'] == 'Transfer as-is']
    print(f"   Transfer as-is quality: {transfer_asis_3['transfer_quality_pct'].values[0]:.1f}%")
    print(f"   Best strategy: {results3.loc[results3['silhouette'].idxmax(), 'strategy']}")
    
    print("\nExperiment 4 (LOW-MODERATE Transferability):")
    transfer_asis_4 = results4[results4['strategy'] == 'Transfer as-is']
    print(f"   Transfer as-is quality: {transfer_asis_4['transfer_quality_pct'].values[0]:.1f}%")
    print(f"   Best strategy: {results4.loc[results4['silhouette'].idxmax(), 'strategy']}")
    
    print("\n📦 DELIVERABLES:")
    print("  ✅ results/experiment3_results.csv")
    print("  ✅ results/experiment4_results.csv")
    print("  ✅ results/experiments_3_4_combined.csv")
    
    print("\n🎯 NEXT STEPS:")
    print("  1. Run: python cross_experiment_analysis.py")
    print("  2. Review: results/cross_experiment_analysis.xlsx")
    print("  3. Write: experiments_3_4_report.pdf")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()