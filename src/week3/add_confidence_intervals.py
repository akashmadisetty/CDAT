#!/usr/bin/env python3
"""
Add Bootstrap Confidence Intervals to Framework
================================================

Adds 95% confidence intervals to transferability scores using bootstrap resampling.

This script patches the TransferLearningFramework to include uncertainty quantification.

Usage:
    Run this once to add confidence interval methods to framework.py
    
Author: Member 3 (Week 5-6 enhancement)
Date: November 2024
"""

import sys
from pathlib import Path

# Instructions for adding to framework.py
PATCH_CODE = '''
# ============================================================================
# ADD THIS TO framework.py (after calculate_transferability method)
# ============================================================================

def calculate_confidence_interval(self, n_bootstrap=1000, confidence_level=0.95, verbose=True):
    """
    Calculate confidence interval for transferability score using bootstrap resampling.
    
    Parameters:
    -----------
    n_bootstrap : int
        Number of bootstrap samples (default: 1000)
    confidence_level : float
        Confidence level (default: 0.95 for 95% CI)
    verbose : bool
        Print progress (default: True)
    
    Returns:
    --------
    dict : {
        'score': mean score,
        'ci_lower': lower bound,
        'ci_upper': upper bound,
        'std_error': standard error,
        'sample_scores': all bootstrap scores
    }
    """
    if verbose:
        print(f"\\n🔄 Calculating {confidence_level*100:.0f}% confidence interval...")
        print(f"   Bootstrap samples: {n_bootstrap}")
    
    bootstrap_scores = []
    
    for i in range(n_bootstrap):
        if verbose and (i+1) % 200 == 0:
            print(f"   Progress: {i+1}/{n_bootstrap}")
        
        # Resample with replacement
        source_sample = self.source_data.sample(n=len(self.source_data), replace=True)
        target_sample = self.target_data.sample(n=len(self.target_data), replace=True)
        
        # Create temporary framework instance
        temp_framework = TransferLearningFramework(
            source_model=self.source_model,
            source_data=source_sample,
            target_data=target_sample,
            use_learned_weights=self.use_learned_weights
        )
        
        # Calculate transferability
        temp_framework.calculate_transferability(verbose=False)
        bootstrap_scores.append(temp_framework.composite_score)
    
    # Calculate statistics
    bootstrap_scores = np.array(bootstrap_scores)
    mean_score = np.mean(bootstrap_scores)
    std_error = np.std(bootstrap_scores)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_scores, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
    
    self.confidence_interval = {
        'score': mean_score,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std_error': std_error,
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap,
        'sample_scores': bootstrap_scores
    }
    
    if verbose:
        print(f"\\n📊 Confidence Interval Results:")
        print(f"   Mean Score: {mean_score:.4f}")
        print(f"   {confidence_level*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   Standard Error: {std_error:.4f}")
        print(f"   Original Score: {self.composite_score:.4f}")
    
    return self.confidence_interval


def get_confidence_interval_summary(self):
    """Get formatted confidence interval summary"""
    if not hasattr(self, 'confidence_interval'):
        return "Confidence interval not calculated. Run calculate_confidence_interval() first."
    
    ci = self.confidence_interval
    return (
        f"Score: {ci['score']:.4f} "
        f"(95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}], "
        f"SE: {ci['std_error']:.4f})"
    )

# ============================================================================
# END OF PATCH
# ============================================================================
'''

SIMPLE_USAGE = '''
# ============================================================================
# SIMPLE USAGE IN CLI.py
# ============================================================================

# After calculating transferability, add this:

if args.with_confidence:
    print("\\n📊 Calculating confidence interval...")
    ci = framework.calculate_confidence_interval(n_bootstrap=1000, verbose=True)
    
    print(f"\\n{Colors.BOLD}Transferability Score with 95% Confidence Interval:{Colors.END}")
    print(f"  Score: {ci['score']:.4f}")
    print(f"  95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print(f"  Standard Error: {ci['std_error']:.4f}")

# Add flag to argparse:
parser.add_argument('--with-confidence', action='store_true',
                   help='Calculate 95% confidence interval (slower, uses bootstrap)')

# ============================================================================
'''

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║              ADD CONFIDENCE INTERVALS TO FRAMEWORK                            ║
║                    Bootstrap-based Uncertainty Quantification                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

This script provides code to add confidence interval calculation to your framework.

📋 What to do:

1. **Manual Addition (Recommended):**
   - Open src/week3/framework.py
   - Find the `calculate_transferability` method
   - Add the two new methods shown below
   
2. **Update CLI (Optional but Nice):**
   - Open src/week3/cli.py  
   - Add the --with-confidence flag
   - Display confidence intervals when requested

""")
    
    print("\n" + "="*80)
    print("CODE TO ADD TO framework.py:")
    print("="*80)
    print(PATCH_CODE)
    
    print("\n" + "="*80)
    print("CODE TO ADD TO cli.py:")
    print("="*80)
    print(SIMPLE_USAGE)
    
    print("\n" + "="*80)
    print("EXAMPLE OUTPUT:")
    print("="*80)
    print("""
🔄 Calculating 95% confidence interval...
   Bootstrap samples: 1000
   Progress: 200/1000
   Progress: 400/1000
   Progress: 600/1000
   Progress: 800/1000
   Progress: 1000/1000

📊 Confidence Interval Results:
   Mean Score: 0.8751
   95% CI: [0.8234, 0.9156]
   Standard Error: 0.0231
   Original Score: 0.8749

✅ Transferability Score with 95% Confidence Interval:
  Score: 0.8751
  95% CI: [0.8234, 0.9156]
  Standard Error: 0.0231
  
INTERPRETATION:
- We are 95% confident the true transferability is between 0.8234 and 0.9156
- Small standard error (0.0231) indicates high precision
- Narrow CI width (0.0922) suggests stable estimate
""")
    
    print("\n" + "="*80)
    print("WHY THIS IS BETTER THAN validate_framework_on_uk_retail.py:")
    print("="*80)
    print("""
✅ Integrated into framework - works everywhere (CLI, notebooks, scripts)
✅ Optional flag --with-confidence - doesn't slow down normal usage
✅ Bootstrap method - scientifically sound, no assumptions about distribution
✅ Accounts for sample size - small samples get wider CIs automatically
✅ User-friendly - just one flag to add

EXAMPLE USAGE:
    
    # Without CI (fast)
    python cli.py --mode builtin --pair 1
    
    # With CI (slower, but more informative)
    python cli.py --mode builtin --pair 1 --with-confidence
    
    # For UK Retail data
    python cli.py --mode rfm --source exp5_uk_source_RFM_scaled.csv \\
                  --target exp5_france_target_RFM_scaled.csv --with-confidence
""")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
1. Copy the two methods to src/week3/framework.py
2. (Optional) Add --with-confidence flag to src/week3/cli.py  
3. Test it:
   cd src/week3
   python cli.py --mode builtin --pair 1 --with-confidence

4. Use in validation (if you want):
   - The validation script can call calculate_confidence_interval()
   - Save CI results to CSV alongside composite_score
   - Include in the markdown report

TIME ESTIMATE:
- Adding to framework.py: 5 minutes (copy-paste)
- Adding to cli.py: 5 minutes (copy-paste + test)
- Total: 10 minutes

DECISION:
Do you want me to:
A) Create the modified framework.py and cli.py files ready to use? ✅
B) Just keep the instructions above? (you manually add)
C) Skip confidence intervals entirely? (100% accuracy already!)

Say A, B, or C!
""")

if __name__ == '__main__':
    main()
