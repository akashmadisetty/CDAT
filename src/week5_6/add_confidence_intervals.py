#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Confidence Intervals to Transferability Predictions
========================================================

Implements bootstrap-based confidence interval calculation for:
1. Transferability scores (95% CI)
2. Individual metrics (MMD, JS Divergence, etc.)
3. Sample-size-aware uncertainty quantification

Author: Member 3 (Week 5-6 Enhancement)
Date: November 2024
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add week3 to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'week3'))

from framework import TransferLearningFramework

# ============================================================================
# CONFIGURATION
# ============================================================================

WEEK5_6_DIR = Path(__file__).parent
RESULTS_DIR = WEEK5_6_DIR / 'validation_results'
RESULTS_DIR.mkdir(exist_ok=True)

N_BOOTSTRAP = 1000  # Number of bootstrap samples
CONFIDENCE_LEVEL = 0.95  # 95% confidence interval
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# UK Retail Experiments
UK_EXPERIMENTS = {
    5: {
        'name': 'UK → France',
        'source_file': 'exp5_uk_source_RFM_scaled.csv',
        'target_file': 'exp5_france_target_RFM_scaled.csv',
    },
    6: {
        'name': 'UK → Germany',
        'source_file': 'exp5_uk_source_RFM_scaled.csv',
        'target_file': 'exp6_germany_target_RFM_scaled.csv',
    },
    7: {
        'name': 'High-Value → Medium-Value',
        'source_file': 'exp7_highvalue_source_RFM_scaled.csv',
        'target_file': 'exp7_mediumvalue_target_RFM_scaled.csv',
    }
}

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL FUNCTIONS
# ============================================================================

def bootstrap_confidence_interval(source_data, target_data, n_bootstrap=1000, confidence=0.95):
    """
    Calculate confidence intervals using bootstrap resampling.
    
    Bootstrap Method:
    1. Resample source and target data WITH replacement
    2. Calculate transferability score for each resample
    3. Compute percentiles for confidence interval
    
    Parameters:
    -----------
    source_data : DataFrame
        Source domain RFM features
    target_data : DataFrame
        Target domain RFM features
    n_bootstrap : int
        Number of bootstrap iterations
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
    --------
    dict with mean, std, CI_lower, CI_upper for each metric
    """
    
    print(f"   🔄 Running {n_bootstrap} bootstrap iterations...")
    
    # Storage for bootstrap results
    composite_scores = []
    mmd_scores = []
    js_scores = []
    corr_scores = []
    
    feature_cols = ['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']
    
    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            print(f"      Progress: {i+1}/{n_bootstrap} iterations...")
        
        # Resample WITH replacement (bootstrap)
        source_boot = source_data.sample(n=len(source_data), replace=True)
        target_boot = target_data.sample(n=len(target_data), replace=True)
        
        try:
            # Create framework instance
            framework = TransferLearningFramework(
                source_model=None,
                source_data=source_boot,
                target_data=target_boot,
                use_learned_weights=True
            )
            
            # Calculate transferability
            framework.calculate_transferability(verbose=False)
            
            # Store results
            metrics = framework.transferability_metrics
            composite_scores.append(framework.composite_score)
            mmd_scores.append(metrics.get('mmd', 0))
            js_scores.append(metrics.get('js_divergence', 0))
            corr_scores.append(metrics.get('correlation_stability', 0))
            
        except Exception as e:
            # Skip failed iterations (very rare)
            continue
    
    print(f"   ✅ Bootstrap complete! Successful iterations: {len(composite_scores)}/{n_bootstrap}")
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    results = {
        'composite_score': {
            'mean': np.mean(composite_scores),
            'std': np.std(composite_scores),
            'ci_lower': np.percentile(composite_scores, lower_percentile),
            'ci_upper': np.percentile(composite_scores, upper_percentile),
            'ci_width': np.percentile(composite_scores, upper_percentile) - np.percentile(composite_scores, lower_percentile)
        },
        'mmd_score': {
            'mean': np.mean(mmd_scores),
            'std': np.std(mmd_scores),
            'ci_lower': np.percentile(mmd_scores, lower_percentile),
            'ci_upper': np.percentile(mmd_scores, upper_percentile),
            'ci_width': np.percentile(mmd_scores, upper_percentile) - np.percentile(mmd_scores, lower_percentile)
        },
        'js_divergence': {
            'mean': np.mean(js_scores),
            'std': np.std(js_scores),
            'ci_lower': np.percentile(js_scores, lower_percentile),
            'ci_upper': np.percentile(js_scores, upper_percentile),
            'ci_width': np.percentile(js_scores, upper_percentile) - np.percentile(js_scores, lower_percentile)
        },
        'correlation_stability': {
            'mean': np.mean(corr_scores),
            'std': np.std(corr_scores),
            'ci_lower': np.percentile(corr_scores, lower_percentile),
            'ci_upper': np.percentile(corr_scores, upper_percentile),
            'ci_width': np.percentile(corr_scores, upper_percentile) - np.percentile(corr_scores, lower_percentile)
        }
    }
    
    return results


def calculate_sample_size_adjusted_ci(source_size, target_size, base_ci_width):
    """
    Adjust confidence interval width based on sample size.
    
    Smaller samples → Wider confidence intervals (more uncertainty)
    
    Rule of thumb:
    - < 100 samples: Increase CI by 30%
    - 100-500 samples: Increase CI by 15%
    - > 500 samples: No adjustment
    
    Parameters:
    -----------
    source_size : int
        Number of source domain samples
    target_size : int
        Number of target domain samples
    base_ci_width : float
        Base confidence interval width from bootstrap
    
    Returns:
    --------
    adjusted_width : float
        Sample-size-adjusted CI width
    """
    
    min_size = min(source_size, target_size)
    
    if min_size < 100:
        adjustment_factor = 1.30  # 30% wider
        reason = "SMALL SAMPLE (< 100)"
    elif min_size < 500:
        adjustment_factor = 1.15  # 15% wider
        reason = "MODERATE SAMPLE (100-500)"
    else:
        adjustment_factor = 1.00  # No adjustment
        reason = "LARGE SAMPLE (> 500)"
    
    adjusted_width = base_ci_width * adjustment_factor
    
    return adjusted_width, reason


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_confidence_interval_analysis():
    """Run bootstrap CI analysis for all experiments"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         CONFIDENCE INTERVAL ANALYSIS - TRANSFERABILITY PREDICTIONS            ║
║                   Bootstrap-Based Uncertainty Quantification                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")
    
    all_results = []
    
    for exp_num in sorted(UK_EXPERIMENTS.keys()):
        config = UK_EXPERIMENTS[exp_num]
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {exp_num}: {config['name']}")
        print(f"{'='*80}")
        
        # Load data
        source_path = WEEK5_6_DIR / config['source_file']
        target_path = WEEK5_6_DIR / config['target_file']
        
        if not source_path.exists() or not target_path.exists():
            print(f"❌ Files not found. Run uk_rfm_generator_FIXED.py first!")
            continue
        
        source_data = pd.read_csv(source_path)
        target_data = pd.read_csv(target_path)
        
        print(f"\n📊 Dataset:")
        print(f"   Source: {len(source_data):,} customers")
        print(f"   Target: {len(target_data):,} customers")
        
        # Calculate point estimate (single run)
        print(f"\n📌 Point Estimate (Original):")
        framework = TransferLearningFramework(
            source_model=None,
            source_data=source_data,
            target_data=target_data,
            use_learned_weights=True
        )
        framework.calculate_transferability(verbose=False)
        point_estimate = framework.composite_score
        print(f"   Composite Score: {point_estimate:.4f}")
        
        # Calculate confidence intervals
        print(f"\n🎲 Bootstrap Analysis:")
        ci_results = bootstrap_confidence_interval(
            source_data, target_data, 
            n_bootstrap=N_BOOTSTRAP, 
            confidence=CONFIDENCE_LEVEL
        )
        
        # Sample size adjustment
        adj_width, adj_reason = calculate_sample_size_adjusted_ci(
            len(source_data), len(target_data),
            ci_results['composite_score']['ci_width']
        )
        
        print(f"\n📊 Confidence Interval Results:")
        print(f"   Composite Score:")
        print(f"      Point Estimate:  {point_estimate:.4f}")
        print(f"      Bootstrap Mean:  {ci_results['composite_score']['mean']:.4f}")
        print(f"      Std Dev:         {ci_results['composite_score']['std']:.4f}")
        print(f"      95% CI:          [{ci_results['composite_score']['ci_lower']:.4f}, {ci_results['composite_score']['ci_upper']:.4f}]")
        print(f"      CI Width:        {ci_results['composite_score']['ci_width']:.4f}")
        
        print(f"\n   Sample Size Adjustment:")
        print(f"      Factor:          {adj_width / ci_results['composite_score']['ci_width']:.2f}x")
        print(f"      Reason:          {adj_reason}")
        print(f"      Adjusted Width:  {adj_width:.4f}")
        
        # Calculate adjusted CI bounds
        mean_score = ci_results['composite_score']['mean']
        adj_ci_lower = mean_score - (adj_width / 2)
        adj_ci_upper = mean_score + (adj_width / 2)
        
        print(f"      Adjusted 95% CI: [{adj_ci_lower:.4f}, {adj_ci_upper:.4f}]")
        
        print(f"\n   Individual Metrics (95% CI):")
        print(f"      MMD:              [{ci_results['mmd_score']['ci_lower']:.4f}, {ci_results['mmd_score']['ci_upper']:.4f}]")
        print(f"      JS Divergence:    [{ci_results['js_divergence']['ci_lower']:.4f}, {ci_results['js_divergence']['ci_upper']:.4f}]")
        print(f"      Correlation Stab: [{ci_results['correlation_stability']['ci_lower']:.4f}, {ci_results['correlation_stability']['ci_upper']:.4f}]")
        
        # Interpretation
        print(f"\n💡 Interpretation:")
        if ci_results['composite_score']['ci_width'] < 0.05:
            print(f"   ✅ HIGH PRECISION: Narrow confidence interval (width < 0.05)")
            print(f"      Prediction is VERY RELIABLE")
        elif ci_results['composite_score']['ci_width'] < 0.10:
            print(f"   ✅ GOOD PRECISION: Moderate confidence interval (width < 0.10)")
            print(f"      Prediction is RELIABLE")
        elif ci_results['composite_score']['ci_width'] < 0.15:
            print(f"   ⚠️  MODERATE PRECISION: Wider confidence interval (width < 0.15)")
            print(f"      Prediction is SOMEWHAT UNCERTAIN")
        else:
            print(f"   ❌ LOW PRECISION: Very wide confidence interval (width > 0.15)")
            print(f"      Prediction has HIGH UNCERTAINTY - Need more data!")
        
        if len(target_data) < 100:
            print(f"\n   ⚠️  Small target sample ({len(target_data)} customers)")
            print(f"      Recommendation: Collect more target data if possible")
        
        # Store results
        result_row = {
            'experiment': exp_num,
            'name': config['name'],
            'source_size': len(source_data),
            'target_size': len(target_data),
            'point_estimate': point_estimate,
            'bootstrap_mean': ci_results['composite_score']['mean'],
            'bootstrap_std': ci_results['composite_score']['std'],
            'ci_lower_95': ci_results['composite_score']['ci_lower'],
            'ci_upper_95': ci_results['composite_score']['ci_upper'],
            'ci_width': ci_results['composite_score']['ci_width'],
            'adj_ci_lower_95': adj_ci_lower,
            'adj_ci_upper_95': adj_ci_upper,
            'adj_ci_width': adj_width,
            'adj_reason': adj_reason,
            'precision_rating': 'HIGH' if ci_results['composite_score']['ci_width'] < 0.05 else
                               'GOOD' if ci_results['composite_score']['ci_width'] < 0.10 else
                               'MODERATE' if ci_results['composite_score']['ci_width'] < 0.15 else 'LOW'
        }
        all_results.append(result_row)
    
    # ========================================================================
    # SUMMARY & COMPARISON
    # ========================================================================
    
    print(f"\n\n{'='*80}")
    print(f"SUMMARY: CONFIDENCE INTERVAL ANALYSIS")
    print(f"{'='*80}\n")
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        print(f"📊 All Experiments with Confidence Intervals:\n")
        display_cols = ['experiment', 'name', 'point_estimate', 'ci_lower_95', 'ci_upper_95', 'ci_width', 'precision_rating']
        print(summary_df[display_cols].to_string(index=False))
        
        # Save results
        output_file = RESULTS_DIR / 'confidence_intervals_analysis.csv'
        summary_df.to_csv(output_file, index=False)
        print(f"\n✅ Detailed results saved to: {output_file}")
        
        # Statistical insights
        print(f"\n{'='*80}")
        print(f"STATISTICAL INSIGHTS")
        print(f"{'='*80}\n")
        
        print(f"Confidence Interval Widths:")
        print(f"  Mean:   {summary_df['ci_width'].mean():.4f}")
        print(f"  Median: {summary_df['ci_width'].median():.4f}")
        print(f"  Min:    {summary_df['ci_width'].min():.4f} (most precise)")
        print(f"  Max:    {summary_df['ci_width'].max():.4f} (least precise)")
        
        print(f"\nPrecision Ratings:")
        for rating in ['HIGH', 'GOOD', 'MODERATE', 'LOW']:
            count = (summary_df['precision_rating'] == rating).sum()
            print(f"  {rating}: {count} experiment(s)")
        
        print(f"\nSample Size vs Uncertainty:")
        for idx, row in summary_df.iterrows():
            uncertainty_pct = (row['ci_width'] / row['point_estimate']) * 100
            print(f"  Exp {row['experiment']}: {row['target_size']:,} target customers → ±{uncertainty_pct:.1f}% uncertainty")
        
        # Generate markdown report
        generate_ci_report(summary_df)
    
    print(f"\n{'='*80}")
    print(f"✅ CONFIDENCE INTERVAL ANALYSIS COMPLETE!")
    print(f"{'='*80}\n")
    print(f"📁 Results saved to: {RESULTS_DIR}")
    print(f"\nKey Takeaways:")
    print(f"  1. Confidence intervals quantify prediction uncertainty")
    print(f"  2. Smaller samples → Wider CIs (more uncertainty)")
    print(f"  3. All predictions include 95% confidence bounds")
    print(f"  4. Use these for risk assessment in transfer decisions")
    print()


def generate_ci_report(summary_df):
    """Generate detailed CI report in markdown"""
    
    report_path = RESULTS_DIR / 'confidence_intervals_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Confidence Interval Analysis Report
## Bootstrap-Based Uncertainty Quantification

**Date:** November 2024  
**Method:** Bootstrap resampling (n={N_BOOTSTRAP} iterations)  
**Confidence Level:** {CONFIDENCE_LEVEL*100:.0f}%

---

## Executive Summary

This report provides **confidence intervals** for all transferability predictions, quantifying the **uncertainty** in our framework's recommendations.

### Why Confidence Intervals Matter

- **Point estimates** (single scores) don't tell the full story
- **Confidence intervals** show the range of plausible values
- **Small samples** (France: 87, Germany: 94) have wider CIs = more uncertainty
- **Decision-making** should account for this uncertainty

---

## Results Summary

| Experiment | Point Estimate | 95% CI | Precision |
|------------|---------------|---------|-----------|
""")
        
        for idx, row in summary_df.iterrows():
            f.write(f"| {row['experiment']}: {row['name']} | {row['point_estimate']:.4f} | [{row['ci_lower_95']:.4f}, {row['ci_upper_95']:.4f}] | {row['precision_rating']} |\n")
        
        f.write(f"""

---

## Detailed Analysis

""")
        
        for idx, row in summary_df.iterrows():
            uncertainty_pct = (row['ci_width'] / row['point_estimate']) * 100
            
            f.write(f"""### Experiment {row['experiment']}: {row['name']}

**Sample Sizes:**
- Source: {row['source_size']:,} customers
- Target: {row['target_size']:,} customers

**Transferability Score:**
- Point Estimate: `{row['point_estimate']:.4f}`
- Bootstrap Mean: `{row['bootstrap_mean']:.4f}`
- Bootstrap Std Dev: `{row['bootstrap_std']:.4f}`

**95% Confidence Interval:**
- CI Bounds: `[{row['ci_lower_95']:.4f}, {row['ci_upper_95']:.4f}]`
- CI Width: `{row['ci_width']:.4f}`
- Relative Uncertainty: `±{uncertainty_pct:.1f}%`

**Sample Size Adjustment:**
- Adjustment Reason: {row['adj_reason']}
- Adjusted CI: `[{row['adj_ci_lower_95']:.4f}, {row['adj_ci_upper_95']:.4f}]`
- Adjusted Width: `{row['adj_ci_width']:.4f}`

**Precision Rating:** **{row['precision_rating']}**

**Interpretation:**
""")
            
            if row['precision_rating'] == 'HIGH':
                f.write(f"- ✅ **Very reliable prediction** - narrow confidence interval indicates high certainty\n")
            elif row['precision_rating'] == 'GOOD':
                f.write(f"- ✅ **Reliable prediction** - moderate confidence interval indicates good certainty\n")
            elif row['precision_rating'] == 'MODERATE':
                f.write(f"- ⚠️ **Somewhat uncertain prediction** - wider confidence interval suggests moderate uncertainty\n")
            else:
                f.write(f"- ❌ **Uncertain prediction** - very wide confidence interval indicates high uncertainty\n")
            
            if row['target_size'] < 100:
                f.write(f"- ⚠️ **Small target sample** ({row['target_size']} customers) contributes to uncertainty\n")
                f.write(f"- 💡 **Recommendation:** Collect more target data if possible to improve precision\n")
            
            f.write(f"\n---\n\n")
        
        f.write(f"""## Statistical Summary

### Confidence Interval Statistics

- **Mean CI Width:** {summary_df['ci_width'].mean():.4f}
- **Median CI Width:** {summary_df['ci_width'].median():.4f}
- **Min CI Width:** {summary_df['ci_width'].min():.4f} (most precise)
- **Max CI Width:** {summary_df['ci_width'].max():.4f} (least precise)

### Precision Distribution

""")
        
        for rating in ['HIGH', 'GOOD', 'MODERATE', 'LOW']:
            count = (summary_df['precision_rating'] == rating).sum()
            f.write(f"- **{rating}:** {count} experiment(s)\n")
        
        f.write(f"""

---

## Methodology

### Bootstrap Resampling

1. **Resample** source and target data WITH replacement
2. **Calculate** transferability score for each resample
3. **Repeat** {N_BOOTSTRAP} times
4. **Compute** percentiles for confidence interval

### Sample Size Adjustment

Small samples have inherently more uncertainty:

- **< 100 samples:** CI width increased by 30%
- **100-500 samples:** CI width increased by 15%
- **> 500 samples:** No adjustment needed

This accounts for sampling variability in small target domains.

---

## Recommendations

1. **Use adjusted CIs** for decision-making with small samples
2. **Consider uncertainty** when choosing transfer strategies
3. **Collect more data** for high-uncertainty predictions (LOW precision)
4. **Trust high-precision predictions** (HIGH rating) with confidence

---

## Conclusion

Confidence intervals provide crucial context for transferability predictions:

- ✅ **Quantifies uncertainty** in a statistically rigorous way
- ✅ **Accounts for sample size** differences across domains
- ✅ **Enables risk-aware** transfer learning decisions
- ✅ **Identifies cases** where more data is needed

**Key Insight:** Not all predictions are equally certain. Use confidence intervals to assess risk before committing to a transfer strategy.

---

*Generated by: add_confidence_intervals.py*  
*Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*Bootstrap Iterations: {N_BOOTSTRAP}*  
*Confidence Level: {CONFIDENCE_LEVEL*100:.0f}%*
""")
    
    print(f"\n✅ CI report saved to: {report_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        run_confidence_interval_analysis()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
