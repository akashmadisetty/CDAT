"""
Run Transfer Learning Experiments for Pairs 1 & 2
Member 1 - Week 3 Deliverable
"""

import pandas as pd
from transfer_experiments import TransferLearningExperiment, calculate_source_baseline
import os

# ========================================
# ONLY CHANGE THIS SECTION IF NEEDED:
# ========================================
EXPERIMENTS = {
    1: {
        'name': 'Cleaning and Household → Foodgrains,Oil & Masala',
        'expected_transfer': 'HIGH',
        'week1_score': 0.892,
        'source_rfm': '../week2/domain_pair1_source_RFM.csv',
        'target_rfm': '../week2/domain_pair1_target_RFM.csv',
        'source_model': '../week2/models/domain_pair1_rfm_kmeans_model.pkl'
    },
    2: {
        'name': 'Snacks & Branded Foods → Fruits & Vegetables',
        'expected_transfer': 'MODERATE',
        'week1_score': 0.763,
        'source_rfm': '../week2/domain_pair2_source_RFM.csv',
        'target_rfm': '../week2/domain_pair2_target_RFM.csv',
        'source_model': '../week2/models/domain_pair2_rfm_kmeans_model.pkl'
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
    print("TRANSFER LEARNING EXPERIMENTS - MEMBER 1 WEEK 3")
    print("Pairs 1 & 2: HIGH and MODERATE Transferability")
    print("🚀"*40)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run Experiment 1
    print("\n" + "🟢"*40)
    print("EXPERIMENT 1: HIGH TRANSFERABILITY (Cleaning and Household → Foodgrains,Oil & Masala)")
    print("🟢"*40)
    results1 = run_experiment(1, EXPERIMENTS[1])
    
    # Run Experiment 2
    print("\n" + "🟡"*40)
    print("EXPERIMENT 2: MODERATE TRANSFERABILITY (Snacks & Branded Foods → Fruits & Vegetables)")
    print("🟡"*40)
    results2 = run_experiment(2, EXPERIMENTS[2])
    
    # Combined analysis
    combined = pd.concat([results1, results2], ignore_index=True)
    combined.to_csv('results/experiments_1_2_combined.csv', index=False)
    
    # Summary
    print("\n" + "="*80)
    print("🎉 EXPERIMENTS COMPLETE!")
    print("="*80)
    
    print("\n📊 QUICK SUMMARY:")
    print("\nExperiment 1 (HIGH Transferability):")
    transfer_asis_1 = results1[results1['strategy'] == 'Transfer as-is']
    print(f"   Transfer as-is quality: {transfer_asis_1['transfer_quality_pct'].values[0]:.1f}%")
    print(f"   Best strategy: {results1.loc[results1['silhouette'].idxmax(), 'strategy']}")
    
    print("\nExperiment 2 (MODERATE Transferability):")
    transfer_asis_2 = results2[results2['strategy'] == 'Transfer as-is']
    print(f"   Transfer as-is quality: {transfer_asis_2['transfer_quality_pct'].values[0]:.1f}%")
    print(f"   Best strategy: {results2.loc[results2['silhouette'].idxmax(), 'strategy']}")
    
    print("\n📦 DELIVERABLES:")
    print("  ✅ results/experiment1_results.csv")
    print("  ✅ results/experiment2_results.csv")
    print("  ✅ results/experiments_1_2_combined.csv")
    
    print("\n🎯 NEXT STEPS:")
    print("  1. Run: python analyze_results_member1.py (for visualizations)")
    print("  2. Write: experiments_1_2_report.pdf")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()