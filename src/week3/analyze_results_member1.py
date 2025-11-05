"""
Analyze and Visualize Transfer Learning Experiment Results
Member 1 - Week 3-4 Deliverable
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_results():
    """Load experiment results"""
    print("\n📂 Loading experiment results...")
    
    try:
        combined = pd.read_csv('results/experiments_1_2_combined.csv')
        print(f"  ✓ Loaded {len(combined)} experiment runs")
        return combined
    except FileNotFoundError:
        print("  ❌ ERROR: Run 'python run_experiments.py' first!")
        return None


def plot_performance_comparison(df):
    """
    Plot 1: Performance comparison across strategies
    """
    print("\n📊 Creating Plot 1: Performance Comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Silhouette scores by strategy
    strategies = df['strategy'].unique()
    pair1_data = df[df['pair_id'] == 1]
    pair2_data = df[df['pair_id'] == 2]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    pair1_silh = [pair1_data[pair1_data['strategy']==s]['silhouette'].values[0] for s in strategies]
    pair2_silh = [pair2_data[pair2_data['strategy']==s]['silhouette'].values[0] for s in strategies]
    
    axes[0].bar(x - width/2, pair1_silh, width, label='Pair 1 (HIGH)', alpha=0.8, color='steelblue')
    axes[0].bar(x + width/2, pair2_silh, width, label='Pair 2 (MODERATE)', alpha=0.8, color='coral')
    
    axes[0].set_xlabel('Strategy', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Clustering Quality by Strategy', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategies, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Transfer quality percentage
    pair1_quality = [pair1_data[pair1_data['strategy']==s]['transfer_quality_pct'].values[0] for s in strategies]
    pair2_quality = [pair2_data[pair2_data['strategy']==s]['transfer_quality_pct'].values[0] for s in strategies]
    
    axes[1].bar(x - width/2, pair1_quality, width, label='Pair 1 (HIGH)', alpha=0.8, color='steelblue')
    axes[1].bar(x + width/2, pair2_quality, width, label='Pair 2 (MODERATE)', alpha=0.8, color='coral')
    axes[1].axhline(y=100, color='green', linestyle='--', label='Train from scratch baseline', alpha=0.5)
    axes[1].axhline(y=85, color='orange', linestyle='--', label='Good transfer threshold', alpha=0.5)
    
    axes[1].set_xlabel('Strategy', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Transfer Quality (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Transfer Quality vs Train-from-Scratch', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(strategies, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/plot1_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/plot1_performance_comparison.png")
    plt.show()


def plot_fine_tuning_curve(df):
    """
    Plot 2: Fine-tuning curve (performance vs % target data)
    """
    print("\n📊 Creating Plot 2: Fine-Tuning Curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract data for fine-tuning strategies
    target_pcts = [0, 10, 20, 50, 100]  # Percentages of target data
    
    pair1_data = df[df['pair_id'] == 1].sort_values('target_data_pct')
    pair2_data = df[df['pair_id'] == 2].sort_values('target_data_pct')
    
    pair1_silh = pair1_data['silhouette'].values
    pair2_silh = pair2_data['silhouette'].values
    
    # Silhouette curve
    axes[0].plot(target_pcts, pair1_silh, 'o-', linewidth=2, markersize=8, 
                label='Pair 1 (HIGH)', color='steelblue')
    axes[0].plot(target_pcts, pair2_silh, 's-', linewidth=2, markersize=8, 
                label='Pair 2 (MODERATE)', color='coral')
    
    axes[0].set_xlabel('Target Data Used (%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Clustering Quality vs Target Data Usage', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(target_pcts)
    
    # Transfer quality curve
    pair1_quality = pair1_data['transfer_quality_pct'].values
    pair2_quality = pair2_data['transfer_quality_pct'].values
    
    axes[1].plot(target_pcts, pair1_quality, 'o-', linewidth=2, markersize=8, 
                label='Pair 1 (HIGH)', color='steelblue')
    axes[1].plot(target_pcts, pair2_quality, 's-', linewidth=2, markersize=8, 
                label='Pair 2 (MODERATE)', color='coral')
    axes[1].axhline(y=85, color='orange', linestyle='--', label='Good transfer (85%)', alpha=0.5)
    
    axes[1].set_xlabel('Target Data Used (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Transfer Quality (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Transfer Quality vs Target Data Usage', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(target_pcts)
    
    plt.tight_layout()
    plt.savefig('results/plot2_fine_tuning_curves.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/plot2_fine_tuning_curves.png")
    plt.show()


def plot_validation_heatmap(df):
    """
    Plot 3: Prediction validation heatmap
    """
    print("\n📊 Creating Plot 3: Prediction Validation...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create matrix for heatmap
    strategies = df['strategy'].unique()
    pairs = ['Pair 1\n(HIGH Expected)', 'Pair 2\n(MODERATE Expected)']
    
    matrix = np.zeros((len(pairs), len(strategies)))
    
    for i, pair_id in enumerate([1, 2]):
        pair_data = df[df['pair_id'] == pair_id]
        for j, strategy in enumerate(strategies):
            silh = pair_data[pair_data['strategy'] == strategy]['silhouette'].values[0]
            matrix[i, j] = silh
    
    # Create heatmap
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.2, vmax=0.5,
                xticklabels=strategies, yticklabels=pairs, cbar_kws={'label': 'Silhouette Score'},
                ax=ax)
    
    ax.set_title('Transfer Learning Performance Heatmap\n(Green = Better Performance)', 
                fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/plot3_validation_heatmap.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/plot3_validation_heatmap.png")
    plt.show()


def plot_cost_benefit_analysis(df):
    """
    Plot 4: Cost-benefit analysis (performance gain vs data requirement)
    """
    print("\n📊 Creating Plot 4: Cost-Benefit Analysis...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each strategy as a point (x=data cost, y=performance)
    for pair_id in [1, 2]:
        pair_data = df[df['pair_id'] == 1].copy()
        pair_name = 'Pair 1 (HIGH)' if pair_id == 1 else 'Pair 2 (MODERATE)'
        color = 'steelblue' if pair_id == 1 else 'coral'
        
        # Data cost = % target data needed
        x = pair_data['target_data_pct'].values
        y = pair_data['silhouette'].values
        
        # Plot points
        ax.scatter(x, y, s=200, alpha=0.6, color=color, label=pair_name, edgecolors='black', linewidth=2)
        
        # Annotate each point
        for idx, row in pair_data.iterrows():
            ax.annotate(row['strategy'], 
                       (row['target_data_pct'], row['silhouette']),
                       textcoords="offset points", xytext=(0,10), ha='center',
                       fontsize=9, fontweight='bold')
    
    # Add zones
    ax.axhspan(0.4, 0.5, alpha=0.1, color='green', label='Excellent (>0.4)')
    ax.axhspan(0.3, 0.4, alpha=0.1, color='yellow', label='Good (0.3-0.4)')
    ax.axhspan(0, 0.3, alpha=0.1, color='red', label='Poor (<0.3)')
    
    ax.set_xlabel('Data Requirement (% of Target Data Needed)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (Silhouette Score)', fontsize=12, fontweight='bold')
    ax.set_title('Cost-Benefit Analysis: Performance vs Data Requirement', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plot4_cost_benefit_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/plot4_cost_benefit_analysis.png")
    plt.show()


def generate_summary_report(df):
    """
    Generate text summary report
    """
    print("\n📝 Generating summary report...")
    
    report = []
    report.append("="*80)
    report.append("TRANSFER LEARNING EXPERIMENTS - SUMMARY REPORT")
    report.append("Member 1 - Week 3-4 Deliverable")
    report.append("="*80)
    
    # Overall statistics
    report.append("\n1. OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Total experiment runs: {len(df)}")
    report.append(f"Domain pairs tested: 2")
    report.append(f"Strategies tested per pair: 5")
    report.append(f"Average silhouette score: {df['silhouette'].mean():.3f}")
    report.append(f"Best strategy overall: {df.groupby('strategy')['silhouette'].mean().idxmax()}")
    
    # Experiment 1 results
    report.append("\n2. EXPERIMENT 1: HIGH TRANSFERABILITY PAIR")
    report.append("-" * 80)
    pair1 = df[df['pair_id'] == 1]
    report.append(f"Pair: {pair1.iloc[0]['pair_name']}")
    report.append(f"Week 1 Prediction: {pair1.iloc[0]['expected_transfer']} (score: {pair1.iloc[0]['week1_score']:.3f})")
    report.append(f"Source model performance: {pair1.iloc[0]['source_silhouette']:.3f}")
    report.append("\nResults by strategy:")
    for _, row in pair1.iterrows():
        report.append(f"  {row['strategy']:<25} Silhouette: {row['silhouette']:.3f}  Quality: {row['transfer_quality_pct']:.1f}%")
    
    # Experiment 2 results
    report.append("\n3. EXPERIMENT 2: MODERATE TRANSFERABILITY PAIR")
    report.append("-" * 80)
    pair2 = df[df['pair_id'] == 2]
    report.append(f"Pair: {pair2.iloc[0]['pair_name']}")
    report.append(f"Week 1 Prediction: {pair2.iloc[0]['expected_transfer']} (score: {pair2.iloc[0]['week1_score']:.3f})")
    report.append(f"Source model performance: {pair2.iloc[0]['source_silhouette']:.3f}")
    report.append("\nResults by strategy:")
    for _, row in pair2.iterrows():
        report.append(f"  {row['strategy']:<25} Silhouette: {row['silhouette']:.3f}  Quality: {row['transfer_quality_pct']:.1f}%")
    
    # Key findings
    report.append("\n4. KEY FINDINGS")
    report.append("-" * 80)
    
    # Finding 1: Transfer as-is performance
    asis_p1 = pair1[pair1['strategy'] == 'Transfer as-is'].iloc[0]
    asis_p2 = pair2[pair2['strategy'] == 'Transfer as-is'].iloc[0]
    
    report.append(f"\nFinding 1: Transfer As-Is Performance")
    report.append(f"  Pair 1: {asis_p1['transfer_quality_pct']:.1f}% of train-from-scratch")
    report.append(f"  Pair 2: {asis_p2['transfer_quality_pct']:.1f}% of train-from-scratch")
    
    if asis_p1['transfer_quality_pct'] >= 85:
        report.append(f"  ✅ Pair 1: HIGH transferability validated!")
    if asis_p2['transfer_quality_pct'] >= 70:
        report.append(f"  ✅ Pair 2: MODERATE transferability validated!")
    
    # Finding 2: Best strategy
    report.append(f"\nFinding 2: Optimal Strategy")
    best_strat_p1 = pair1.loc[pair1['silhouette'].idxmax(), 'strategy']
    best_strat_p2 = pair2.loc[pair2['silhouette'].idxmax(), 'strategy']
    report.append(f"  Pair 1: {best_strat_p1} (Silh: {pair1['silhouette'].max():.3f})")
    report.append(f"  Pair 2: {best_strat_p2} (Silh: {pair2['silhouette'].max():.3f})")
    
    # Finding 3: Fine-tuning benefit
    report.append(f"\nFinding 3: Fine-Tuning Analysis")
    ft10_p1 = pair1[pair1['target_data_pct'] == 10].iloc[0]
    ft10_p2 = pair2[pair2['target_data_pct'] == 10].iloc[0]
    
    improvement_p1 = ft10_p1['silhouette'] - asis_p1['silhouette']
    improvement_p2 = ft10_p2['silhouette'] - asis_p2['silhouette']
    
    report.append(f"  Pair 1: 10% fine-tuning improves by {improvement_p1:.3f} points")
    report.append(f"  Pair 2: 10% fine-tuning improves by {improvement_p2:.3f} points")
    
    # Recommendations
    report.append("\n5. RECOMMENDATIONS")
    report.append("-" * 80)
    report.append("\nFor HIGH transferability pairs (like Pair 1):")
    report.append("  ✓ Transfer as-is achieves >85% quality - recommended approach")
    report.append("  ✓ Fine-tuning with 10% improves marginally - use if data available")
    
    report.append("\nFor MODERATE transferability pairs (like Pair 2):")
    report.append("  ⚠ Transfer as-is achieves 70-85% quality - acceptable but suboptimal")
    report.append("  ✓ Fine-tuning with 10-20% target data recommended")
    report.append("  ✓ Consider training from scratch if >50% target data available")
    
    report.append("\n" + "="*80)
    
    # Save report
    report_text = "\n".join(report)
    with open('results/experiments_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("  ✓ Saved: results/experiments_summary_report.txt")
    
    # Print to console
    print("\n" + report_text)
    
    return report_text


def main():
    """
    Main analysis pipeline
    """
    print("\n" + "🚀"*40)
    print("TRANSFER LEARNING EXPERIMENTS - ANALYSIS & VISUALIZATION")
    print("🚀"*40)
    
    # Load results
    df = load_results()
    if df is None:
        return
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    plot_performance_comparison(df)
    plot_fine_tuning_curve(df)
    plot_validation_heatmap(df)
    plot_cost_benefit_analysis(df)
    
    # Generate summary report
    generate_summary_report(df)
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\n📦 FILES CREATED:")
    print("  ✅ results/plot1_performance_comparison.png")
    print("  ✅ results/plot2_fine_tuning_curves.png")
    print("  ✅ results/plot3_validation_heatmap.png")
    print("  ✅ results/plot4_cost_benefit_analysis.png")
    print("  ✅ results/experiments_summary_report.txt")
    
    print("\n🎯 NEXT STEPS:")
    print("  1. Review all visualizations in results/ folder")
    print("  2. Read experiments_summary_report.txt")
    print("  3. Write experiments_1_2_report.pdf using these findings")
    print("  4. Present results to team")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()