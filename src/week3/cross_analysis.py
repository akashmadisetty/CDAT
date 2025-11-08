"""
Cross-Analysis Script: Compare ALL Domain Pairs
Performs statistical analysis to answer key research questions

Key Questions:
1. Does predicted transferability correlate with actual transfer performance?
2. When is fine-tuning most beneficial?
3. What's the optimal amount of target data for fine-tuning?
4. Are there patterns across high/moderate/low transferability pairs?
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os

from experiment_config import DOMAIN_PAIRS, PATHS


class CrossAnalysis:
    """Perform cross-experiment analysis across all domain pairs"""
    
    def __init__(self, results_file=None):
        if results_file is None:
            results_file = f"{PATHS['results_dir']}/ALL_EXPERIMENTS_RESULTS.csv"
        
        self.df = pd.read_csv(results_file)
        self.output_dir = PATHS['results_dir']
        
        # Add transferability scores to results
        self.df['transferability_score'] = self.df['pair_number'].apply(
            lambda x: DOMAIN_PAIRS[x]['transferability_score']
        )
        self.df['expected_category'] = self.df['pair_number'].apply(
            lambda x: DOMAIN_PAIRS[x]['expected_transferability']
        )
        
        print(f"✓ Loaded {len(self.df)} experiment results for cross-analysis")
    
    def question_1_transferability_correlation(self):
        """Q1: Does predicted transferability correlate with actual performance?"""
        print("\n" + "="*80)
        print("QUESTION 1: Transferability Score vs Actual Performance")
        print("="*80)
        
        # Get zero-shot performance for each pair
        zero_shot = self.df[self.df['test_number'] == 1].copy()
        
        # Calculate correlations
        pearson_r, pearson_p = pearsonr(zero_shot['transferability_score'], 
                                        zero_shot['silhouette_score'])
        spearman_r, spearman_p = spearmanr(zero_shot['transferability_score'], 
                                           zero_shot['silhouette_score'])
        
        print(f"\n📊 Correlation Analysis:")
        print(f"  Pearson correlation:  {pearson_r:.4f} (p={pearson_p:.4f})")
        print(f"  Spearman correlation: {spearman_r:.4f} (p={spearman_p:.4f})")
        
        if pearson_p < 0.05:
            print(f"  ✓ Statistically significant correlation! (p < 0.05)")
        else:
            print(f"  ✗ Not statistically significant (p >= 0.05)")
        
        # Show pair-by-pair comparison
        print(f"\n📋 Pair-by-Pair Breakdown:")
        print("-" * 80)
        comparison = zero_shot[['pair_number', 'pair_name', 'transferability_score', 
                               'silhouette_score']].sort_values('transferability_score', 
                                                                ascending=False)
        comparison['rank_predicted'] = range(1, len(comparison) + 1)
        comparison = comparison.sort_values('silhouette_score', ascending=False)
        comparison['rank_actual'] = range(1, len(comparison) + 1)
        comparison['rank_diff'] = abs(comparison['rank_predicted'] - comparison['rank_actual'])
        
        print(comparison[['pair_number', 'transferability_score', 'silhouette_score', 
                         'rank_predicted', 'rank_actual', 'rank_diff']].to_string(index=False))
        
        avg_rank_diff = comparison['rank_diff'].mean()
        print(f"\n  Average rank difference: {avg_rank_diff:.2f}")
        print(f"  (Lower is better - means predictions match reality)")
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'avg_rank_diff': avg_rank_diff,
            'comparison_df': comparison
        }
    
    def question_2_when_finetune_helps(self):
        """Q2: When is fine-tuning most beneficial?"""
        print("\n" + "="*80)
        print("QUESTION 2: When Does Fine-Tuning Help Most?")
        print("="*80)
        
        improvements = []
        
        for pair_num in sorted(self.df['pair_number'].unique()):
            df_pair = self.df[self.df['pair_number'] == pair_num]
            pair_info = DOMAIN_PAIRS[pair_num]
            
            zero_shot = df_pair[df_pair['test_number'] == 1]['silhouette_score'].values[0]
            finetune_10 = df_pair[df_pair['test_number'] == 2]['silhouette_score'].values[0]
            finetune_20 = df_pair[df_pair['test_number'] == 3]['silhouette_score'].values[0]
            finetune_50 = df_pair[df_pair['test_number'] == 4]['silhouette_score'].values[0]
            from_scratch = df_pair[df_pair['test_number'] == 5]['silhouette_score'].values[0]
            
            improvements.append({
                'pair_number': pair_num,
                'transferability': pair_info['transferability_score'],
                'category': pair_info['expected_transferability'],
                'zero_shot': zero_shot,
                'improvement_10': finetune_10 - zero_shot,
                'improvement_20': finetune_20 - zero_shot,
                'improvement_50': finetune_50 - zero_shot,
                'from_scratch': from_scratch,
                'best_finetune': max(finetune_10, finetune_20, finetune_50),
                'beats_scratch': max(finetune_10, finetune_20, finetune_50) > from_scratch
            })
        
        df_imp = pd.DataFrame(improvements)
        
        print(f"\n📊 Fine-tuning Improvements by Transferability Category:")
        print("-" * 80)
        
        for category in ['HIGH', 'MODERATE-HIGH', 'MODERATE', 'LOW']:
            category_data = df_imp[df_imp['category'] == category]
            if len(category_data) > 0:
                avg_imp_10 = category_data['improvement_10'].mean()
                avg_imp_20 = category_data['improvement_20'].mean()
                avg_imp_50 = category_data['improvement_50'].mean()
                
                print(f"\n{category} Transferability ({len(category_data)} pairs):")
                print(f"  Avg improvement with 10% data: {avg_imp_10:+.4f}")
                print(f"  Avg improvement with 20% data: {avg_imp_20:+.4f}")
                print(f"  Avg improvement with 50% data: {avg_imp_50:+.4f}")
        
        # Correlation: Lower transferability = More improvement?
        corr_10, p_10 = pearsonr(df_imp['transferability'], df_imp['improvement_10'])
        corr_20, p_20 = pearsonr(df_imp['transferability'], df_imp['improvement_20'])
        corr_50, p_50 = pearsonr(df_imp['transferability'], df_imp['improvement_50'])
        
        print(f"\n📈 Correlation: Transferability vs Fine-tuning Benefit")
        print("-" * 80)
        print(f"  With 10% data: r={corr_10:+.4f} (p={p_10:.4f})")
        print(f"  With 20% data: r={corr_20:+.4f} (p={p_20:.4f})")
        print(f"  With 50% data: r={corr_50:+.4f} (p={p_50:.4f})")
        
        if corr_10 < -0.5:
            print(f"\n  ✓ Strong negative correlation: Low transferability pairs benefit MORE from fine-tuning!")
        
        return df_imp
    
    def question_3_optimal_data_amount(self):
        """Q3: What's the optimal amount of target data for fine-tuning?"""
        print("\n" + "="*80)
        print("QUESTION 3: Optimal Amount of Target Data")
        print("="*80)
        
        optimal_results = []
        
        for pair_num in sorted(self.df['pair_number'].unique()):
            df_pair = self.df[self.df['pair_number'] == pair_num]
            pair_info = DOMAIN_PAIRS[pair_num]
            
            finetune_tests = df_pair[df_pair['test_number'].isin([2, 3, 4])]
            best_test = finetune_tests.loc[finetune_tests['silhouette_score'].idxmax()]
            
            optimal_results.append({
                'pair_number': pair_num,
                'category': pair_info['expected_transferability'],
                'optimal_percentage': int(best_test['target_data_percentage']),
                'best_score': best_test['silhouette_score'],
                'zero_shot_score': df_pair[df_pair['test_number'] == 1]['silhouette_score'].values[0]
            })
        
        df_optimal = pd.DataFrame(optimal_results)
        
        print(f"\n📊 Optimal Fine-tuning Data by Pair:")
        print("-" * 80)
        print(df_optimal[['pair_number', 'category', 'optimal_percentage', 
                         'best_score', 'zero_shot_score']].to_string(index=False))
        
        print(f"\n📈 Distribution of Optimal Percentages:")
        print("-" * 80)
        for pct in [10, 20, 50]:
            count = len(df_optimal[df_optimal['optimal_percentage'] == pct])
            print(f"  {pct}% data: {count} pairs ({count/len(df_optimal)*100:.1f}%)")
        
        # Average by category
        print(f"\n📋 Average Optimal % by Transferability:")
        print("-" * 80)
        for category in df_optimal['category'].unique():
            cat_data = df_optimal[df_optimal['category'] == category]
            avg_pct = cat_data['optimal_percentage'].mean()
            print(f"  {category}: {avg_pct:.1f}%")
        
        return df_optimal
    
    def question_4_transferability_patterns(self):
        """Q4: Are there patterns across high/moderate/low transferability pairs?"""
        print("\n" + "="*80)
        print("QUESTION 4: Patterns by Transferability Level")
        print("="*80)
        
        patterns = []
        
        for pair_num in sorted(self.df['pair_number'].unique()):
            df_pair = self.df[self.df['pair_number'] == pair_num]
            pair_info = DOMAIN_PAIRS[pair_num]
            
            patterns.append({
                'pair_number': pair_num,
                'category': pair_info['expected_transferability'],
                'transferability': pair_info['transferability_score'],
                'zero_shot': df_pair[df_pair['test_number'] == 1]['silhouette_score'].values[0],
                'best_finetune': df_pair[df_pair['test_number'].isin([2,3,4])]['silhouette_score'].max(),
                'from_scratch': df_pair[df_pair['test_number'] == 5]['silhouette_score'].values[0],
                'total_training_time': df_pair['training_time'].sum()
            })
        
        df_patterns = pd.DataFrame(patterns)
        df_patterns['transfer_benefit'] = df_patterns['zero_shot'] - df_patterns['from_scratch']
        df_patterns['finetune_benefit'] = df_patterns['best_finetune'] - df_patterns['zero_shot']
        
        print(f"\n📊 Summary by Transferability Category:")
        print("-" * 80)
        
        summary = df_patterns.groupby('category').agg({
            'zero_shot': ['mean', 'std'],
            'best_finetune': ['mean', 'std'],
            'from_scratch': ['mean', 'std'],
            'transfer_benefit': ['mean', 'std'],
            'finetune_benefit': ['mean', 'std']
        }).round(4)
        
        print(summary)
        
        # Key insights
        print(f"\n💡 Key Insights:")
        print("-" * 80)
        
        # When is zero-shot good enough?
        high_transfer = df_patterns[df_patterns['category'].str.contains('HIGH')]
        if len(high_transfer) > 0:
            avg_zero = high_transfer['zero_shot'].mean()
            avg_scratch = high_transfer['from_scratch'].mean()
            print(f"  • HIGH transferability pairs:")
            print(f"    - Zero-shot avg: {avg_zero:.4f}")
            print(f"    - From scratch avg: {avg_scratch:.4f}")
            if avg_zero >= avg_scratch * 0.95:
                print(f"    ✓ Zero-shot is competitive! (≥95% of from-scratch)")
        
        # When is fine-tuning essential?
        low_transfer = df_patterns[df_patterns['category'] == 'LOW']
        if len(low_transfer) > 0:
            avg_finetune_benefit = low_transfer['finetune_benefit'].mean()
            print(f"\n  • LOW transferability pairs:")
            print(f"    - Avg improvement from fine-tuning: {avg_finetune_benefit:+.4f}")
            if avg_finetune_benefit > 0.05:
                print(f"    ✓ Fine-tuning provides significant benefit!")
        
        return df_patterns
    
    def generate_cross_analysis_report(self, save=True):
        """Generate complete cross-analysis report"""
        print("\n" + "="*80)
        print("CROSS-DOMAIN TRANSFER LEARNING: COMPLETE ANALYSIS")
        print("="*80)
        
        # Run all analyses
        q1_results = self.question_1_transferability_correlation()
        q2_results = self.question_2_when_finetune_helps()
        q3_results = self.question_3_optimal_data_amount()
        q4_results = self.question_4_transferability_patterns()
        
        # Generate visualizations
        self._plot_cross_analysis(q1_results, q2_results, q3_results, q4_results, save=save)
        
        # Save detailed report
        if save:
            report_path = f"{self.output_dir}/CROSS_ANALYSIS_REPORT.txt"
            with open(report_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("TRANSFER LEARNING FRAMEWORK: CROSS-ANALYSIS REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write("RESEARCH QUESTIONS:\n")
                f.write("1. Does predicted transferability correlate with actual performance?\n")
                f.write("2. When is fine-tuning most beneficial?\n")
                f.write("3. What's the optimal amount of target data?\n")
                f.write("4. What patterns emerge across transferability levels?\n\n")
                
                f.write("="*80 + "\n")
                f.write("KEY FINDINGS:\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Q1: Transferability Correlation\n")
                f.write(f"   Pearson r = {q1_results['pearson_r']:.4f} (p={q1_results['pearson_p']:.4f})\n")
                f.write(f"   Average rank difference = {q1_results['avg_rank_diff']:.2f}\n\n")
                
                f.write(f"Q2: Fine-tuning Benefits\n")
                f.write(f"   [See detailed breakdown above]\n\n")
                
                f.write(f"Q3: Optimal Data Amount\n")
                f.write(q3_results.to_string(index=False))
                f.write("\n\n")
                
                f.write(f"Q4: Transferability Patterns\n")
                f.write(q4_results.to_string(index=False))
                f.write("\n\n")
            
            print(f"\n✓ Saved detailed report: {report_path}")
        
        print("\n" + "="*80)
        print("CROSS-ANALYSIS COMPLETE!")
        print("="*80)
    
    def _plot_cross_analysis(self, q1_results, q2_results, q3_results, q4_results, save=True):
        """Create comprehensive cross-analysis visualizations"""
        viz_dir = PATHS['visualizations_dir']
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create a 2x2 subplot
        fig = plt.figure(figsize=(18, 12))
        
        # Plot 1: Transferability vs Performance (Q1)
        ax1 = plt.subplot(2, 2, 1)
        comparison = q1_results['comparison_df']
        ax1.scatter(comparison['transferability_score'], comparison['silhouette_score'], 
                   s=150, alpha=0.6, c=comparison['pair_number'], cmap='viridis')
        for idx, row in comparison.iterrows():
            ax1.annotate(f"P{row['pair_number']}", 
                        (row['transferability_score'], row['silhouette_score']),
                        fontsize=10, ha='center')
        ax1.set_xlabel('Predicted Transferability Score', fontsize=11)
        ax1.set_ylabel('Actual Zero-Shot Performance', fontsize=11)
        ax1.set_title(f'Q1: Predicted vs Actual (r={q1_results["pearson_r"]:.3f})', 
                     fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement by Transferability (Q2)
        ax2 = plt.subplot(2, 2, 2)
        x_pos = np.arange(len(q2_results))
        width = 0.25
        ax2.bar(x_pos - width, q2_results['improvement_10'], width, label='10% data', alpha=0.8)
        ax2.bar(x_pos, q2_results['improvement_20'], width, label='20% data', alpha=0.8)
        ax2.bar(x_pos + width, q2_results['improvement_50'], width, label='50% data', alpha=0.8)
        ax2.set_xlabel('Domain Pair', fontsize=11)
        ax2.set_ylabel('Improvement from Fine-tuning', fontsize=11)
        ax2.set_title('Q2: Fine-tuning Benefits by Data Amount', fontweight='bold', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"P{i}" for i in q2_results['pair_number']])
        ax2.legend()
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Optimal Data Distribution (Q3)
        ax3 = plt.subplot(2, 2, 3)
        optimal_counts = q3_results['optimal_percentage'].value_counts().sort_index()
        ax3.bar(optimal_counts.index, optimal_counts.values, color=['green', 'blue', 'orange'])
        ax3.set_xlabel('Optimal Fine-tuning Data %', fontsize=11)
        ax3.set_ylabel('Number of Pairs', fontsize=11)
        ax3.set_title('Q3: Distribution of Optimal Data Amounts', fontweight='bold', fontsize=12)
        ax3.set_xticks([10, 20, 50])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Performance by Category (Q4)
        ax4 = plt.subplot(2, 2, 4)
        categories = q4_results.groupby('category').agg({
            'zero_shot': 'mean',
            'best_finetune': 'mean',
            'from_scratch': 'mean'
        })
        x_pos = np.arange(len(categories))
        width = 0.25
        ax4.bar(x_pos - width, categories['zero_shot'], width, label='Zero-shot', alpha=0.8)
        ax4.bar(x_pos, categories['best_finetune'], width, label='Best Fine-tune', alpha=0.8)
        ax4.bar(x_pos + width, categories['from_scratch'], width, label='From Scratch', alpha=0.8)
        ax4.set_xlabel('Transferability Category', fontsize=11)
        ax4.set_ylabel('Avg Silhouette Score', fontsize=11)
        ax4.set_title('Q4: Performance by Transferability Level', fontweight='bold', fontsize=12)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories.index, rotation=15, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            output_path = f"{viz_dir}/cross_analysis_summary.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved cross-analysis visualization: {output_path}")
        
        plt.show()


if __name__ == "__main__":
    analyzer = CrossAnalysis()
    analyzer.generate_cross_analysis_report(save=True)
