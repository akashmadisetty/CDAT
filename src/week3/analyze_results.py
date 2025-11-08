"""
Analysis and Visualization Script for Transfer Learning Experiments
Generates plots and statistical analysis of experiment results

Usage:
    python analyze_results.py                    # Analyze all results
    python analyze_results.py --pair 1           # Analyze specific pair
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from experiment_config import DOMAIN_PAIRS, PATHS

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ResultsAnalyzer:
    """Analyze and visualize experiment results"""
    
    def __init__(self, results_file=None):
        if results_file is None:
            results_file = f"{PATHS['results_dir']}/ALL_EXPERIMENTS_RESULTS.csv"
        
        self.results_file = results_file
        self.df = pd.read_csv(results_file)
        self.viz_dir = PATHS['visualizations_dir']
        os.makedirs(self.viz_dir, exist_ok=True)
        
        print(f"✓ Loaded {len(self.df)} experiment results")
        print(f"  Pairs: {self.df['pair_number'].nunique()}")
        print(f"  Tests per pair: {self.df.groupby('pair_number').size().values}")
    
    def plot_pair_performance(self, pair_number, save=True):
        """Plot performance metrics across all 5 tests for a single pair"""
        df_pair = self.df[self.df['pair_number'] == pair_number]
        pair_info = DOMAIN_PAIRS[pair_number]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{pair_info['name']}\nTransferability: {pair_info['transferability_score']:.4f}", 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Silhouette Score
        ax1 = axes[0, 0]
        ax1.plot(df_pair['test_number'], df_pair['silhouette_score'], 
                marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Test Number')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score (Higher is Better)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(df_pair['test_number'])
        ax1.set_xticklabels([f"T{i}\n{p}%" for i, p in 
                            zip(df_pair['test_number'], df_pair['target_data_percentage'])])
        
        # Plot 2: Davies-Bouldin Index
        ax2 = axes[0, 1]
        ax2.plot(df_pair['test_number'], df_pair['davies_bouldin_index'], 
                marker='s', linewidth=2, markersize=8, color='coral')
        ax2.set_xlabel('Test Number')
        ax2.set_ylabel('Davies-Bouldin Index')
        ax2.set_title('Davies-Bouldin Index (Lower is Better)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(df_pair['test_number'])
        ax2.set_xticklabels([f"T{i}\n{p}%" for i, p in 
                            zip(df_pair['test_number'], df_pair['target_data_percentage'])])
        
        # Plot 3: Calinski-Harabasz Score
        ax3 = axes[1, 0]
        ax3.plot(df_pair['test_number'], df_pair['calinski_harabasz'], 
                marker='^', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Test Number')
        ax3.set_ylabel('Calinski-Harabasz Score')
        ax3.set_title('Calinski-Harabasz Score (Higher is Better)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(df_pair['test_number'])
        ax3.set_xticklabels([f"T{i}\n{p}%" for i, p in 
                            zip(df_pair['test_number'], df_pair['target_data_percentage'])])
        
        # Plot 4: Training Time
        ax4 = axes[1, 1]
        ax4.bar(df_pair['test_number'], df_pair['training_time'], 
               color='purple', alpha=0.7)
        ax4.set_xlabel('Test Number')
        ax4.set_ylabel('Training Time (seconds)')
        ax4.set_title('Training Time')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_xticks(df_pair['test_number'])
        ax4.set_xticklabels([f"T{i}\n{p}%" for i, p in 
                            zip(df_pair['test_number'], df_pair['target_data_percentage'])])
        
        plt.tight_layout()
        
        if save:
            output_path = f"{self.viz_dir}/pair{pair_number}_performance.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {output_path}")
        
        plt.show()
        return fig
    
    def plot_all_pairs_comparison(self, metric='silhouette_score', save=True):
        """Compare a specific metric across all pairs and tests"""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        for pair_num in sorted(self.df['pair_number'].unique()):
            df_pair = self.df[self.df['pair_number'] == pair_num]
            pair_info = DOMAIN_PAIRS[pair_num]
            
            label = f"Pair {pair_num}: {pair_info['expected_transferability']}"
            ax.plot(df_pair['target_data_percentage'], df_pair[metric], 
                   marker='o', linewidth=2, markersize=6, label=label)
        
        ax.set_xlabel('Target Data Used for Fine-tuning (%)', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} Across All Domain Pairs', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 10, 20, 50, 100])
        
        plt.tight_layout()
        
        if save:
            output_path = f"{self.viz_dir}/all_pairs_{metric}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {output_path}")
        
        plt.show()
        return fig
    
    def plot_transferability_correlation(self, save=True):
        """Plot correlation between predicted transferability and actual performance"""
        # Calculate improvement from zero-shot to best fine-tuned
        improvements = []
        
        for pair_num in sorted(self.df['pair_number'].unique()):
            df_pair = self.df[self.df['pair_number'] == pair_num]
            pair_info = DOMAIN_PAIRS[pair_num]
            
            zero_shot_score = df_pair[df_pair['test_number'] == 1]['silhouette_score'].values[0]
            best_finetune_score = df_pair[df_pair['test_number'].isin([2, 3, 4])]['silhouette_score'].max()
            
            improvement = best_finetune_score - zero_shot_score
            
            improvements.append({
                'pair_number': pair_num,
                'pair_name': pair_info['name'],
                'transferability_score': pair_info['transferability_score'],
                'zero_shot_performance': zero_shot_score,
                'best_finetune_performance': best_finetune_score,
                'improvement': improvement
            })
        
        df_corr = pd.DataFrame(improvements)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Transferability vs Zero-Shot Performance
        ax1 = axes[0]
        ax1.scatter(df_corr['transferability_score'], df_corr['zero_shot_performance'], 
                   s=100, alpha=0.6)
        for idx, row in df_corr.iterrows():
            ax1.annotate(f"P{row['pair_number']}", 
                        (row['transferability_score'], row['zero_shot_performance']),
                        fontsize=9, ha='center')
        
        # Add correlation coefficient
        corr = df_corr['transferability_score'].corr(df_corr['zero_shot_performance'])
        ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax1.transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax1.set_xlabel('Predicted Transferability Score', fontsize=12)
        ax1.set_ylabel('Zero-Shot Silhouette Score', fontsize=12)
        ax1.set_title('Predicted Transferability vs Actual Performance', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Transferability vs Improvement from Fine-tuning
        ax2 = axes[1]
        ax2.scatter(df_corr['transferability_score'], df_corr['improvement'], 
                   s=100, alpha=0.6, color='coral')
        for idx, row in df_corr.iterrows():
            ax2.annotate(f"P{row['pair_number']}", 
                        (row['transferability_score'], row['improvement']),
                        fontsize=9, ha='center')
        
        # Add correlation coefficient
        corr2 = df_corr['transferability_score'].corr(df_corr['improvement'])
        ax2.text(0.05, 0.95, f'Correlation: {corr2:.3f}', 
                transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_xlabel('Predicted Transferability Score', fontsize=12)
        ax2.set_ylabel('Improvement from Fine-tuning', fontsize=12)
        ax2.set_title('Does Low Transferability = More Improvement?', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save:
            output_path = f"{self.viz_dir}/transferability_correlation.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {output_path}")
        
        plt.show()
        return fig, df_corr
    
    def plot_heatmap_all_pairs(self, save=True):
        """Create heatmap of all metrics across pairs and tests"""
        # Pivot for silhouette scores
        pivot_silhouette = self.df.pivot(index='pair_number', 
                                         columns='test_number', 
                                         values='silhouette_score')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_silhouette, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0.5, ax=ax, cbar_kws={'label': 'Silhouette Score'})
        
        ax.set_xlabel('Test Number (0%, 10%, 20%, 50%, 100%)', fontsize=12)
        ax.set_ylabel('Domain Pair', fontsize=12)
        ax.set_title('Silhouette Scores: All Pairs × All Tests', fontsize=14, fontweight='bold')
        
        # Add pair names as y-tick labels
        pair_labels = [f"P{i}: {DOMAIN_PAIRS[i]['expected_transferability']}" 
                      for i in sorted(self.df['pair_number'].unique())]
        ax.set_yticklabels(pair_labels, rotation=0)
        
        plt.tight_layout()
        
        if save:
            output_path = f"{self.viz_dir}/heatmap_all_results.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {output_path}")
        
        plt.show()
        return fig
    
    def generate_summary_statistics(self, save=True):
        """Generate summary statistics table"""
        summary = []
        
        for pair_num in sorted(self.df['pair_number'].unique()):
            df_pair = self.df[self.df['pair_number'] == pair_num]
            pair_info = DOMAIN_PAIRS[pair_num]
            
            summary.append({
                'Pair': pair_num,
                'Name': pair_info['name'],
                'Expected_Transferability': pair_info['expected_transferability'],
                'Score': pair_info['transferability_score'],
                'Zero_Shot_Silhouette': df_pair[df_pair['test_number'] == 1]['silhouette_score'].values[0],
                'Best_Finetune_Silhouette': df_pair[df_pair['test_number'].isin([2,3,4])]['silhouette_score'].max(),
                'From_Scratch_Silhouette': df_pair[df_pair['test_number'] == 5]['silhouette_score'].values[0],
                'Improvement_Finetune': df_pair[df_pair['test_number'].isin([2,3,4])]['silhouette_score'].max() - 
                                       df_pair[df_pair['test_number'] == 1]['silhouette_score'].values[0],
                'Total_Training_Time': df_pair['training_time'].sum()
            })
        
        df_summary = pd.DataFrame(summary)
        
        print("\n" + "="*100)
        print("EXPERIMENT SUMMARY STATISTICS")
        print("="*100)
        print(df_summary.to_string(index=False))
        print("="*100)
        
        if save:
            output_path = f"{PATHS['results_dir']}/summary_statistics.csv"
            df_summary.to_csv(output_path, index=False)
            print(f"\n✓ Saved summary: {output_path}")
        
        return df_summary
    
    def generate_all_visualizations(self):
        """Generate all visualizations at once"""
        print("\n" + "="*80)
        print("GENERATING ALL VISUALIZATIONS")
        print("="*80)
        
        # Individual pair plots
        print("\n1. Individual pair performance plots...")
        for pair_num in range(1, 8):
            self.plot_pair_performance(pair_num, save=True)
        
        # Comparison plots
        print("\n2. All pairs comparison plots...")
        self.plot_all_pairs_comparison('silhouette_score', save=True)
        self.plot_all_pairs_comparison('davies_bouldin_index', save=True)
        self.plot_all_pairs_comparison('calinski_harabasz', save=True)
        
        # Correlation plot
        print("\n3. Transferability correlation plot...")
        self.plot_transferability_correlation(save=True)
        
        # Heatmap
        print("\n4. Results heatmap...")
        self.plot_heatmap_all_pairs(save=True)
        
        # Summary statistics
        print("\n5. Summary statistics...")
        self.generate_summary_statistics(save=True)
        
        print("\n" + "="*80)
        print("ALL VISUALIZATIONS COMPLETED!")
        print(f"Saved to: {self.viz_dir}")
        print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--pair', type=int, help='Analyze specific pair (1-7)')
    parser.add_argument('--file', type=str, help='Path to results CSV file')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(results_file=args.file)
    
    if args.pair:
        # Analyze specific pair
        analyzer.plot_pair_performance(args.pair)
        analyzer.generate_summary_statistics()
    else:
        # Generate all visualizations
        analyzer.generate_all_visualizations()
