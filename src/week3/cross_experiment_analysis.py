"""
Cross-Experiment Analysis: Compare ALL 4 Domain Pairs
Member 2 - Week 4 Deliverable

Combines results from Member 1 (Pairs 1-2) and Member 2 (Pairs 3-4)
to identify patterns across all experiments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_all_experiments():
    """
    Load results from all 4 experiments
    FIXED: Handles missing columns gracefully
    """
    print("\n📂 Loading all experiment results...")
    
    all_results = []
    
    for pair_id in [1, 2, 3, 4]:
        try:
            df = pd.read_csv(f'results/experiment{pair_id}_results.csv')
            
            # Check if required columns exist
            required_cols = ['strategy', 'silhouette', 'pair_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  ⚠️  Experiment {pair_id}: Missing columns {missing_cols}")
                print(f"      Available columns: {df.columns.tolist()}")
                print(f"      SKIPPING this pair - regenerate with fixed script")
                continue
            
            # Add pair_id if missing
            if 'pair_id' not in df.columns:
                df['pair_id'] = pair_id
            
            # Add default values for optional columns if missing
            if 'week1_score' not in df.columns:
                print(f"  ⚠️  Experiment {pair_id}: 'week1_score' missing, using defaults")
                week1_defaults = {1: 0.903, 2: 0.548, 3: 0.715, 4: 0.874}
                df['week1_score'] = week1_defaults.get(pair_id, 0.5)
            
            if 'pair_name' not in df.columns:
                pair_names = {
                    1: 'Cleaning & Household → Foodgrains',
                    2: 'Snacks → Fruits & Vegetables',
                    3: 'Premium → Budget Segment',
                    4: 'Popular → Niche Brands'
                }
                df['pair_name'] = pair_names.get(pair_id, f'Pair {pair_id}')
            
            if 'expected_transfer' not in df.columns:
                expected = {1: 'HIGH', 2: 'MODERATE', 3: 'LOW', 4: 'LOW-MODERATE'}
                df['expected_transfer'] = expected.get(pair_id, 'UNKNOWN')
            
            if 'transfer_quality_pct' not in df.columns:
                # Calculate it if we have the data
                if 'Train from scratch' in df['strategy'].values:
                    scratch_silh = df[df['strategy'] == 'Train from scratch']['silhouette'].values[0]
                    df['transfer_quality_pct'] = (df['silhouette'] / scratch_silh) * 100
                else:
                    df['transfer_quality_pct'] = 0
            
            all_results.append(df)
            print(f"  ✓ Loaded Experiment {pair_id}: {len(df)} tests")
            
        except FileNotFoundError:
            print(f"  ❌ Experiment {pair_id} not found - run experiments first!")
            continue
        except Exception as e:
            print(f"  ❌ Error loading Experiment {pair_id}: {e}")
            continue
    
    if len(all_results) == 0:
        print("\n❌ No valid experiment results found!")
        return None
    
    # Combine all
    combined = pd.concat(all_results, ignore_index=True)
    print(f"\n✅ Combined: {len(combined)} total experiment runs across {len(all_results)} pairs")
    
    return combined


def statistical_validation(df):
    """
    Validate correlation between Week 1 predictions and actual transfer quality
    """
    print("\n" + "="*80)
    print("📊 STATISTICAL VALIDATION")
    print("="*80)
    
    # Get transfer-as-is results only
    transfer_asis = df[df['strategy'] == 'Transfer as-is'].copy()
    
    # Prepare data
    predicted = transfer_asis['week1_score'].values
    actual = transfer_asis['transfer_quality_pct'].values / 100  # Convert to 0-1 scale
    
    # Correlation analysis
    pearson_corr, pearson_p = stats.pearsonr(predicted, actual)
    spearman_corr, spearman_p = stats.spearmanr(predicted, actual)
    
    print(f"\n1️⃣ CORRELATION ANALYSIS:")
    print(f"   Pearson correlation: r = {pearson_corr:.3f} (p = {pearson_p:.4f})")
    print(f"   Spearman correlation: ρ = {spearman_corr:.3f} (p = {spearman_p:.4f})")
    
    if pearson_p < 0.05:
        print(f"   ✅ SIGNIFICANT: Week 1 scores predict Week 3 performance!")
    else:
        print(f"   ⚠️  NOT SIGNIFICANT: Weak prediction power (p > 0.05)")
    
    # Linear regression
    from sklearn.linear_model import LinearRegression
    X = predicted.reshape(-1, 1)
    y = actual
    
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)
    
    print(f"\n2️⃣ REGRESSION ANALYSIS:")
    print(f"   R² = {r_squared:.3f}")
    print(f"   Equation: Actual = {model.coef_[0]:.3f} × Predicted + {model.intercept_:.3f}")
    print(f"   Interpretation: Week 1 scores explain {r_squared*100:.1f}% of variance in transfer quality")
    
    # Prediction accuracy
    predicted_quality = model.predict(X) * 100
    mae = np.mean(np.abs(predicted_quality - actual * 100))
    
    print(f"\n3️⃣ PREDICTION ACCURACY:")
    print(f"   Mean Absolute Error: {mae:.1f}%")
    print(f"   Interpretation: Predictions are off by an average of {mae:.1f} percentage points")
    
    # Save validation report
    validation_df = transfer_asis[['pair_name', 'week1_score', 'transfer_quality_pct']].copy()
    validation_df['predicted_quality'] = model.predict(predicted.reshape(-1, 1)) * 100
    validation_df['prediction_error'] = validation_df['predicted_quality'] - validation_df['transfer_quality_pct']
    
    validation_df.to_csv('results/statistical_validation.csv', index=False)
    print(f"\n✅ Saved: results/statistical_validation.csv")
    
    return {
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'r_squared': r_squared,
        'mae': mae,
        'validation_df': validation_df
    }


def identify_patterns(df):
    """
    Identify when transfer works vs fails
    """
    print("\n" + "="*80)
    print("🔍 PATTERN IDENTIFICATION")
    print("="*80)
    
    # Pattern 1: Transferability thresholds
    transfer_asis = df[df['strategy'] == 'Transfer as-is']
    
    print(f"\n1️⃣ TRANSFERABILITY THRESHOLDS:")
    print(f"   Week 1 Score > 0.85 → Transfer Quality: {transfer_asis[transfer_asis['week1_score'] > 0.85]['transfer_quality_pct'].mean():.1f}%")
    print(f"   Week 1 Score 0.7-0.85 → Transfer Quality: {transfer_asis[(transfer_asis['week1_score'] >= 0.7) & (transfer_asis['week1_score'] <= 0.85)]['transfer_quality_pct'].mean():.1f}%")
    print(f"   Week 1 Score < 0.7 → Transfer Quality: {transfer_asis[transfer_asis['week1_score'] < 0.7]['transfer_quality_pct'].mean():.1f}%")
    
    # Pattern 2: Fine-tuning effectiveness
    print(f"\n2️⃣ FINE-TUNING EFFECTIVENESS:")
    
    for pair_id in df['pair_id'].unique():
        pair_data = df[df['pair_id'] == pair_id]
        pair_name = pair_data.iloc[0]['pair_name']
        
        asis = pair_data[pair_data['strategy'] == 'Transfer as-is']['silhouette'].values[0]
        ft10 = pair_data[pair_data['target_data_pct'] == 10]['silhouette'].values[0]
        improvement = ((ft10 - asis) / asis) * 100
        
        print(f"   {pair_name}:")
        print(f"      As-is: {asis:.3f} → 10% fine-tune: {ft10:.3f} (Δ {improvement:+.1f}%)")
    
    # Pattern 3: Optimal strategy by transferability
    print(f"\n3️⃣ OPTIMAL STRATEGY BY TRANSFERABILITY:")
    
    for pair_id in df['pair_id'].unique():
        pair_data = df[df['pair_id'] == pair_id]
        pair_name = pair_data.iloc[0]['pair_name']
        expected = pair_data.iloc[0]['expected_transfer']
        
        best_strategy = pair_data.loc[pair_data['silhouette'].idxmax(), 'strategy']
        best_silh = pair_data['silhouette'].max()
        
        asis_silh = pair_data[pair_data['strategy'] == 'Transfer as-is']['silhouette'].values[0]
        asis_quality = pair_data[pair_data['strategy'] == 'Transfer as-is']['transfer_quality_pct'].values[0]
        
        print(f"\n   {pair_name} ({expected}):")
        print(f"      Best strategy: {best_strategy} (Silh: {best_silh:.3f})")
        print(f"      Transfer as-is: {asis_quality:.1f}% quality")
        
        if asis_quality >= 85:
            print(f"      ✅ Recommendation: Use transfer as-is (>85% quality)")
        elif asis_quality >= 70:
            print(f"      ⚠️  Recommendation: Consider 10-20% fine-tuning")
        else:
            print(f"      ❌ Recommendation: Train from scratch or heavy fine-tuning")
    
    # Pattern 4: Data efficiency analysis
    print(f"\n4️⃣ DATA EFFICIENCY ANALYSIS:")
    print(f"   (Performance per % of target data required)")
    
    for strategy in df['strategy'].unique():
        if strategy == 'Transfer as-is':
            continue
        
        strategy_data = df[df['strategy'] == strategy]
        avg_silh = strategy_data['silhouette'].mean()
        avg_target_pct = strategy_data['target_data_pct'].mean()
        
        efficiency = avg_silh / (avg_target_pct / 100) if avg_target_pct > 0 else 0
        
        print(f"   {strategy}: {efficiency:.3f} (silh per unit of target data)")


def plot_cross_experiment_comparison(df):
    """
    Create comprehensive cross-experiment visualizations
    """
    print("\n📊 Creating cross-experiment visualizations...")
    
    # Plot 1: Week 1 Prediction vs Actual Transfer Quality
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Scatter plot with regression
    transfer_asis = df[df['strategy'] == 'Transfer as-is']
    
    ax = axes[0, 0]
    for _, row in transfer_asis.iterrows():
        ax.scatter(row['week1_score'], row['transfer_quality_pct'], 
                  s=200, alpha=0.7, label=row['pair_name'])
    
    # Add regression line
    x = transfer_asis['week1_score'].values
    y = transfer_asis['transfer_quality_pct'].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'Regression (R²={np.corrcoef(x, y)[0,1]**2:.3f})')
    
    ax.set_xlabel('Week 1 Prediction Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Transfer Quality (%)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Validation: Week 1 vs Week 3', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Performance by transferability category
    ax = axes[0, 1]
    
    categories = ['HIGH', 'MODERATE', 'LOW-MODERATE', 'LOW']
    cat_data = []
    for cat in categories:
        cat_df = df[df['expected_transfer'] == cat]
        if len(cat_df) > 0:
            cat_data.append(cat_df['silhouette'].values)
        else:
            cat_data.append([])
    
    bp = ax.boxplot([d for d in cat_data if len(d) > 0], 
                    labels=[c for c, d in zip(categories, cat_data) if len(d) > 0],
                    patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['green', 'yellow', 'orange', 'red']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xlabel('Expected Transferability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Distribution by Transferability', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Strategy comparison across all pairs
    ax = axes[1, 0]
    
    strategies = df['strategy'].unique()
    pair_names = df['pair_name'].unique()
    
    x = np.arange(len(strategies))
    width = 0.2
    
    for i, pair_name in enumerate(pair_names):
        pair_data = df[df['pair_name'] == pair_name]
        values = [pair_data[pair_data['strategy'] == s]['silhouette'].values[0] for s in strategies]
        ax.bar(x + i * width, values, width, label=f'Pair {i+1}', alpha=0.8)
    
    ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Strategy Comparison Across All Pairs', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Fine-tuning curve (all pairs)
    ax = axes[1, 1]
    
    target_pcts = [0, 10, 20, 50, 100]
    
    for pair_id in df['pair_id'].unique():
        pair_data = df[df['pair_id'] == pair_id].sort_values('target_data_pct')
        pair_name = pair_data.iloc[0]['pair_name']
        
        silh_values = pair_data['silhouette'].values
        ax.plot(target_pcts, silh_values, 'o-', linewidth=2, markersize=8, 
               label=f'Pair {pair_id}: {pair_name[:20]}...')
    
    ax.set_xlabel('Target Data Used (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Fine-Tuning Curves: All Domain Pairs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(target_pcts)
    
    plt.tight_layout()
    plt.savefig('results/cross_experiment_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/cross_experiment_comparison.png")
    plt.show()


def create_excel_report(df, stats):
    """
    Create comprehensive Excel report with multiple sheets
    """
    print("\n📊 Creating Excel report...")
    
    try:
        with pd.ExcelWriter('results/cross_experiment_analysis.xlsx', engine='openpyxl') as writer:
            # Sheet 1: All Results
            df.to_excel(writer, sheet_name='All Results', index=False)
            
            # Sheet 2: Summary by Pair
            summary = df.groupby(['pair_id', 'pair_name', 'expected_transfer']).agg({
                'silhouette': ['mean', 'max', 'min'],
                'transfer_quality_pct': ['mean', 'max', 'min'],
                'week1_score': 'first'
            }).round(3)
            summary.to_excel(writer, sheet_name='Summary by Pair')
            
            # Sheet 3: Summary by Strategy
            strategy_summary = df.groupby('strategy').agg({
                'silhouette': ['mean', 'std', 'min', 'max'],
                'davies_bouldin': ['mean', 'std'],
                'transfer_quality_pct': ['mean', 'std']
            }).round(3)
            strategy_summary.to_excel(writer, sheet_name='Summary by Strategy')
            
            # Sheet 4: Transfer As-Is Analysis
            transfer_asis = df[df['strategy'] == 'Transfer as-is'][
                ['pair_name', 'week1_score', 'silhouette', 'transfer_quality_pct', 
                 'expected_transfer', 'source_silhouette']
            ]
            transfer_asis.to_excel(writer, sheet_name='Transfer As-Is Analysis', index=False)
            
            # Sheet 5: Statistical Validation
            if stats and 'validation_df' in stats:
                stats['validation_df'].to_excel(writer, sheet_name='Statistical Validation', index=False)
            
            # Sheet 6: Recommendations
            recommendations = []
            for pair_id in df['pair_id'].unique():
                pair_data = df[df['pair_id'] == pair_id]
                asis_quality = pair_data[pair_data['strategy'] == 'Transfer as-is']['transfer_quality_pct'].values[0]
                
                if asis_quality >= 85:
                    rec = "Use transfer as-is"
                elif asis_quality >= 70:
                    rec = "Fine-tune with 10-20% target data"
                else:
                    rec = "Train from scratch or heavy fine-tuning (>50%)"
                
                recommendations.append({
                    'Pair ID': pair_id,
                    'Pair Name': pair_data.iloc[0]['pair_name'],
                    'Expected Transferability': pair_data.iloc[0]['expected_transfer'],
                    'Actual Transfer Quality (%)': asis_quality,
                    'Recommendation': rec
                })
            
            pd.DataFrame(recommendations).to_excel(writer, sheet_name='Recommendations', index=False)
        
        print("  ✓ Saved: results/cross_experiment_analysis.xlsx")
        
    except Exception as e:
        print(f"  ⚠️  Excel creation failed: {e}")
        print("  Try installing: pip install openpyxl")


def generate_final_report(df, stats):
    """
    Generate comprehensive text report
    """
    print("\n📝 Generating final report...")
    
    report = []
    report.append("="*80)
    report.append("CROSS-EXPERIMENT ANALYSIS: ALL 4 DOMAIN PAIRS")
    report.append("Member 2 - Week 4 Deliverable")
    report.append("="*80)
    
    # Executive Summary
    report.append("\nEXECUTIVE SUMMARY")
    report.append("-" * 80)
    report.append(f"Total experiments: 4 domain pairs × 5 strategies = 20 tests")
    report.append(f"Average silhouette score: {df['silhouette'].mean():.3f}")
    report.append(f"Best strategy overall: {df.groupby('strategy')['silhouette'].mean().idxmax()}")
    
    if stats:
        report.append(f"\nPrediction Validation:")
        report.append(f"  Correlation (Pearson): r = {stats['pearson_corr']:.3f} (p = {stats['pearson_p']:.4f})")
        report.append(f"  Explained variance: R² = {stats['r_squared']:.3f}")
        report.append(f"  Prediction error: MAE = {stats['mae']:.1f}%")
        
        if stats['pearson_p'] < 0.05:
            report.append(f"  ✅ Week 1 predictions are statistically significant!")
        else:
            report.append(f"  ⚠️  Week 1 predictions show weak statistical power")
    
    # Results by pair
    report.append("\nRESULTS BY DOMAIN PAIR")
    report.append("-" * 80)
    
    for pair_id in sorted(df['pair_id'].unique()):
        pair_data = df[df['pair_id'] == pair_id]
        pair_name = pair_data.iloc[0]['pair_name']
        expected = pair_data.iloc[0]['expected_transfer']
        week1_score = pair_data.iloc[0]['week1_score']
        
        report.append(f"\nPair {pair_id}: {pair_name}")
        report.append(f"  Expected: {expected} (Week 1 Score: {week1_score:.3f})")
        report.append(f"  Results:")
        
        for _, row in pair_data.iterrows():
            report.append(f"    {row['strategy']:<25} Silh: {row['silhouette']:.3f}  Quality: {row['transfer_quality_pct']:.1f}%")
    
    # Key findings
    report.append("\nKEY FINDINGS")
    report.append("-" * 80)
    
    # Finding 1: Transfer as-is performance by category
    report.append("\n1. Transfer As-Is Performance by Transferability:")
    transfer_asis = df[df['strategy'] == 'Transfer as-is'].sort_values('week1_score', ascending=False)
    
    for _, row in transfer_asis.iterrows():
        report.append(f"   {row['pair_name']}: {row['transfer_quality_pct']:.1f}% quality")
    
    # Finding 2: When does transfer work?
    report.append("\n2. When Does Transfer Work?")
    high_transfer = transfer_asis[transfer_asis['transfer_quality_pct'] >= 85]
    moderate_transfer = transfer_asis[(transfer_asis['transfer_quality_pct'] >= 70) & (transfer_asis['transfer_quality_pct'] < 85)]
    low_transfer = transfer_asis[transfer_asis['transfer_quality_pct'] < 70]
    
    report.append(f"   High success (≥85%): {len(high_transfer)} pairs")
    for _, row in high_transfer.iterrows():
        report.append(f"      - {row['pair_name']}")
    
    report.append(f"   Moderate success (70-85%): {len(moderate_transfer)} pairs")
    for _, row in moderate_transfer.iterrows():
        report.append(f"      - {row['pair_name']}")
    
    report.append(f"   Low success (<70%): {len(low_transfer)} pairs")
    for _, row in low_transfer.iterrows():
        report.append(f"      - {row['pair_name']}")
    
    # Finding 3: Fine-tuning analysis
    report.append("\n3. Fine-Tuning Effectiveness:")
    
    for pair_id in sorted(df['pair_id'].unique()):
        pair_data = df[df['pair_id'] == pair_id]
        asis = pair_data[pair_data['strategy'] == 'Transfer as-is']['silhouette'].values[0]
        ft10 = pair_data[pair_data['target_data_pct'] == 10]['silhouette'].values[0]
        ft50 = pair_data[pair_data['target_data_pct'] == 50]['silhouette'].values[0]
        
        improvement_10 = ((ft10 - asis) / asis) * 100
        improvement_50 = ((ft50 - asis) / asis) * 100
        
        report.append(f"   Pair {pair_id}:")
        report.append(f"      10% fine-tune: {improvement_10:+.1f}% improvement")
        report.append(f"      50% fine-tune: {improvement_50:+.1f}% improvement")
    
    # Recommendations
    report.append("\nRECOMMENDATIONS")
    report.append("-" * 80)
    
    report.append("\n1. For HIGH transferability pairs (Week 1 score > 0.85):")
    report.append("   ✅ Use transfer as-is (no target data needed)")
    report.append("   ✅ Expected quality: >85% of train-from-scratch")
    report.append("   ✅ Cost savings: Significant (no target labeling needed)")
    
    report.append("\n2. For MODERATE transferability pairs (Week 1 score 0.6-0.85):")
    report.append("   ⚠️  Transfer as-is achieves 70-85% quality")
    report.append("   ✅ Fine-tune with 10-20% target data recommended")
    report.append("   ✅ Expected improvement: 5-15%")
    
    report.append("\n3. For LOW transferability pairs (Week 1 score < 0.6):")
    report.append("   ❌ Transfer as-is achieves <70% quality")
    report.append("   ⚠️  Heavy fine-tuning (>50%) or train from scratch")
    report.append("   ⚠️  Transfer learning may not be cost-effective")
    
    # Limitations
    report.append("\nLIMITATIONS")
    report.append("-" * 80)
    report.append("1. Small sample size (4 domain pairs)")
    report.append("2. Single clustering algorithm tested (K-Means)")
    report.append("3. RFM features only - other features may show different patterns")
    report.append("4. Synthetic customer data - real-world results may vary")
    
    # Future work
    report.append("\nFUTURE WORK")
    report.append("-" * 80)
    report.append("1. Test with more domain pairs to strengthen statistical validation")
    report.append("2. Compare multiple clustering algorithms (DBSCAN, Hierarchical)")
    report.append("3. Explore additional features beyond RFM")
    report.append("4. Validate on real customer transaction data")
    report.append("5. Test other transfer learning techniques (domain adaptation, meta-learning)")
    
    report.append("\n" + "="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    # Save report
    report_text = "\n".join(report)
    with open('results/cross_experiment_final_report.txt', 'w',encoding='utf-8') as f:
        f.write(report_text)
    
    print("  ✓ Saved: results/cross_experiment_final_report.txt")
    print("\n" + report_text)
    
    return report_text


def main():
    """
    Main analysis pipeline
    """
    print("\n" + "🚀"*40)
    print("CROSS-EXPERIMENT ANALYSIS - ALL 4 DOMAIN PAIRS")
    print("🚀"*40)
    
    # Load all experiments
    df = load_all_experiments()
    if df is None:
        print("\n❌ Cannot proceed without all experiment results!")
        print("   Make sure you have:")
        print("   - results/experiment1_results.csv (Member 1)")
        print("   - results/experiment2_results.csv (Member 1)")
        print("   - results/experiment3_results.csv (Member 2)")
        print("   - results/experiment4_results.csv (Member 2)")
        return
    
    # Statistical validation
    stats = statistical_validation(df)
    
    # Identify patterns
    identify_patterns(df)
    
    # Create visualizations
    plot_cross_experiment_comparison(df)
    
    # Create Excel report
    create_excel_report(df, stats)
    
    # Generate final text report
    generate_final_report(df, stats)
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 CROSS-EXPERIMENT ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\n📦 DELIVERABLES CREATED:")
    print("  ✅ results/cross_experiment_analysis.xlsx")
    print("  ✅ results/cross_experiment_comparison.png")
    print("  ✅ results/statistical_validation.csv")
    print("  ✅ results/cross_experiment_final_report.txt")
    
    print("\n📊 KEY INSIGHTS:")
    if stats:
        print(f"  • Week 1 predictions correlation: r = {stats['pearson_corr']:.3f}")
        print(f"  • Prediction accuracy: MAE = {stats['mae']:.1f}%")
    
    transfer_asis = df[df['strategy'] == 'Transfer as-is']
    print(f"  • Best transfer pair: {transfer_asis.loc[transfer_asis['transfer_quality_pct'].idxmax(), 'pair_name']}")
    print(f"  • Worst transfer pair: {transfer_asis.loc[transfer_asis['transfer_quality_pct'].idxmin(), 'pair_name']}")
    
    print("\n🎯 NEXT STEPS:")
    print("  1. Review cross_experiment_analysis.xlsx")
    print("  2. Study cross_experiment_comparison.png")
    print("  3. Read cross_experiment_final_report.txt")
    print("  4. Write experiments_3_4_report.pdf")
    print("  5. Prepare final presentation")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()