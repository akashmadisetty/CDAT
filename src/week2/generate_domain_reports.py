"""
Complete Domain-Wise RFM Analysis Report Generator
Generates comprehensive reports for each domain with actionable insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")


class DomainReportGenerator:
    """
    Generate comprehensive domain-wise analysis reports
    """
    
    def __init__(self, results_dir='results', plots_dir='plots'):
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        
    def generate_all_reports(self):
        """
        Generate reports for all domains
        """
        # Load summary data
        summary_file = f'{self.results_dir}/improved_baseline_performance.csv'
        if not os.path.exists(summary_file):
            summary_file = f'{self.results_dir}/baseline_performance.csv'
        
        if not os.path.exists(summary_file):
            print("❌ ERROR: No performance summary file found!")
            print("   Please run the training script first.")
            return
        
        df_summary = pd.read_csv(summary_file)
        
        print("\n" + "="*80)
        print("📊 DOMAIN-WISE RFM SEGMENTATION ANALYSIS REPORT")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Create reports directory
        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate report for each domain
        for _, row in df_summary.iterrows():
            domain_id = row['domain_id']
            self.generate_domain_report(domain_id, row, reports_dir)
        
        # Generate comparative summary
        self.generate_comparative_summary(df_summary, reports_dir)
        
        print(f"\n✅ All reports generated in '{reports_dir}/' directory")
        
    def generate_domain_report(self, domain_id, summary_row, reports_dir):
        """
        Generate detailed report for a single domain
        """
        print(f"\n{'='*80}")
        print(f"📄 Generating Report: {summary_row['domain_name']}")
        print(f"{'='*80}")
        
        # Load segment profiles
        profile_file = f'{self.results_dir}/{domain_id}_segment_profiles.csv'
        if not os.path.exists(profile_file):
            print(f"⚠️  Skipping {domain_id}: Profile file not found")
            return
        
        df_profiles = pd.read_csv(profile_file, index_col=0)
        
        # Load customer segments
        segment_file = f'{self.results_dir}/{domain_id}_customer_segments.csv'
        df_segments = pd.read_csv(segment_file) if os.path.exists(segment_file) else None
        
        # Create report
        report = self._create_domain_report_text(domain_id, summary_row, df_profiles, df_segments)
        
        # Save report
        report_file = f'{reports_dir}/{domain_id}_analysis_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Saved: {report_file}")
        
        # Generate visualizations
        self._create_domain_dashboard(domain_id, summary_row, df_profiles, reports_dir)
        
    def _create_domain_report_text(self, domain_id, summary_row, df_profiles, df_segments):
        """
        Create detailed text report for a domain
        """
        report = []
        report.append("="*80)
        report.append(f"RFM CUSTOMER SEGMENTATION ANALYSIS REPORT")
        report.append(f"Domain: {summary_row['domain_name']}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        
        # Executive Summary
        report.append("\n" + "="*80)
        report.append("📊 EXECUTIVE SUMMARY")
        report.append("="*80)
        
        report.append(f"\nDomain Type: {summary_row.get('transferability', 'N/A')}")
        report.append(f"Total Customers Analyzed: {int(summary_row['n_customers']):,}")
        report.append(f"Segments Identified: {int(summary_row['optimal_k'])}")
        
        # Overall quality assessment
        sil_score = summary_row['silhouette_score']
        db_score = summary_row['davies_bouldin_index']
        
        report.append(f"\n{'─'*80}")
        report.append("SEGMENTATION QUALITY:")
        report.append(f"{'─'*80}")
        
        if sil_score > 0.5:
            quality = "EXCELLENT ✓✓✓"
            quality_desc = "Segments are very well-separated and distinct."
        elif sil_score > 0.35:
            quality = "GOOD ✓✓"
            quality_desc = "Segments are clearly defined with good separation."
        elif sil_score > 0.25:
            quality = "ACCEPTABLE ✓"
            quality_desc = "Segments are moderately separated, acceptable for use."
        else:
            quality = "POOR ✗"
            quality_desc = "Segments overlap significantly, consider alternative approaches."
        
        report.append(f"\nOverall Quality: {quality}")
        report.append(f"Assessment: {quality_desc}")
        report.append(f"\nKey Metrics:")
        report.append(f"  • Silhouette Score: {sil_score:.4f} (Range: -1 to 1, Higher is better)")
        report.append(f"  • Davies-Bouldin Index: {db_score:.4f} (Lower is better)")
        report.append(f"  • Calinski-Harabasz Score: {summary_row['calinski_harabasz_score']:.2f} (Higher is better)")
        
        # Interpretation
        report.append(f"\n{'─'*80}")
        report.append("WHAT THIS MEANS:")
        report.append(f"{'─'*80}")
        
        if sil_score > 0.35 and db_score < 1.5:
            report.append("\n✓ The customer segments are WELL-DEFINED and ACTIONABLE")
            report.append("✓ Each segment has distinct characteristics")
            report.append("✓ Marketing strategies can be effectively targeted")
            report.append("✓ High confidence in segment assignments")
            report.append("\nRECOMMENDATION: Proceed with segment-based marketing campaigns")
        elif sil_score > 0.25 and db_score < 2.0:
            report.append("\n⚠ The customer segments are MODERATELY DEFINED")
            report.append("⚠ Some overlap exists between segments")
            report.append("⚠ Consider additional features or different k value")
            report.append("⚠ Monitor segment performance closely")
            report.append("\nRECOMMENDATION: Test with pilot campaigns before full rollout")
        else:
            report.append("\n✗ The customer segments are POORLY DEFINED")
            report.append("✗ Significant overlap between segments")
            report.append("✗ Segment assignments may be unreliable")
            report.append("✗ Consider alternative segmentation approaches")
            report.append("\nRECOMMENDATION: Re-evaluate segmentation strategy or collect more data")
        
        # Segment Analysis
        report.append(f"\n\n{'='*80}")
        report.append("🎯 DETAILED SEGMENT ANALYSIS")
        report.append("="*80)
        
        # Sort by value if available
        if 'Value_Score' in df_profiles.columns:
            df_profiles_sorted = df_profiles.sort_values('Value_Score', ascending=False)
        else:
            df_profiles_sorted = df_profiles
        
        total_customers = int(summary_row['n_customers'])
        
        for idx, row in df_profiles_sorted.iterrows():
            segment_name = row.get('Segment_Name', f'Segment {idx}')
            segment_size = int(row['Recency_count'])
            segment_pct = (segment_size / total_customers) * 100
            
            report.append(f"\n{'─'*80}")
            report.append(f"SEGMENT: {segment_name}")
            report.append(f"{'─'*80}")
            
            report.append(f"\nSize: {segment_size:,} customers ({segment_pct:.1f}% of total)")
            
            if 'Value_Score' in row:
                value_score = row['Value_Score']
                report.append(f"Customer Value Score: {value_score:.1f}/100")
                
                if value_score >= 70:
                    value_rating = "HIGH VALUE ★★★"
                elif value_score >= 50:
                    value_rating = "MEDIUM VALUE ★★"
                else:
                    value_rating = "LOW VALUE ★"
                report.append(f"Value Rating: {value_rating}")
            
            report.append(f"\nRFM Characteristics:")
            report.append(f"  • Recency: {row['Recency_mean']:.1f} days (median: {row['Recency_median']:.1f})")
            report.append(f"  • Frequency: {row['Frequency_mean']:.1f} purchases (median: {row['Frequency_median']:.1f})")
            report.append(f"  • Monetary: ₹{row['Monetary_mean']:.2f} (median: ₹{row['Monetary_median']:.2f})")
            
            # Behavioral interpretation
            report.append(f"\nBehavioral Profile:")
            self._add_segment_interpretation(report, segment_name, row)
            
            # Marketing recommendations
            report.append(f"\n💡 MARKETING RECOMMENDATIONS:")
            self._add_marketing_recommendations(report, segment_name, row, segment_pct)
        
        # Business Impact Analysis
        report.append(f"\n\n{'='*80}")
        report.append("💰 BUSINESS IMPACT ANALYSIS")
        report.append("="*80)
        
        self._add_business_impact(report, df_profiles_sorted, total_customers)
        
        # Transfer Learning Assessment
        report.append(f"\n\n{'='*80}")
        report.append("🔄 TRANSFER LEARNING ASSESSMENT")
        report.append("="*80)
        
        self._add_transfer_learning_assessment(report, summary_row, sil_score, db_score)
        
        # Risks and Limitations
        report.append(f"\n\n{'='*80}")
        report.append("⚠️  RISKS & LIMITATIONS")
        report.append("="*80)
        
        self._add_risks_and_limitations(report, summary_row, df_profiles_sorted)
        
        # Action Items
        report.append(f"\n\n{'='*80}")
        report.append("✅ RECOMMENDED ACTION ITEMS")
        report.append("="*80)
        
        self._add_action_items(report, summary_row, sil_score, db_score)
        
        # Footer
        report.append(f"\n{'='*80}")
        report.append("END OF REPORT")
        report.append("="*80)
        
        return "\n".join(report)
    
    def _add_segment_interpretation(self, report, segment_name, row):
        """
        Add behavioral interpretation for a segment
        """
        r = row['Recency_mean']
        f = row['Frequency_mean']
        m = row['Monetary_mean']
        
        if 'Champion' in segment_name or '🌟' in segment_name:
            report.append("  → Best customers: Recent, frequent, high-value purchases")
            report.append("  → Strong engagement and loyalty")
            report.append("  → Low churn risk")
        elif 'Loyal' in segment_name or '💎' in segment_name:
            report.append("  → Regular customers with consistent engagement")
            report.append("  → Good purchase frequency")
            report.append("  → Reliable revenue stream")
        elif 'At Risk' in segment_name or '⚠️' in segment_name:
            report.append("  → Previously valuable customers showing warning signs")
            report.append("  → Haven't purchased recently despite good history")
            report.append("  → HIGH PRIORITY for retention campaigns")
        elif 'Hibernating' in segment_name or '😴' in segment_name:
            report.append("  → Inactive customers, long time since last purchase")
            report.append("  → Low engagement across all metrics")
            report.append("  → Consider win-back or re-engagement campaigns")
        elif 'New' in segment_name or '🆕' in segment_name:
            report.append("  → Recent customers with limited purchase history")
            report.append("  → High potential for growth")
            report.append("  → Focus on onboarding and early engagement")
        elif 'Promising' in segment_name or '🌱' in segment_name:
            report.append("  → Emerging customers showing positive signs")
            report.append("  → Good initial engagement")
            report.append("  → Opportunity to convert to Champions")
        elif 'Big Spender' in segment_name or '🐋' in segment_name:
            report.append("  → High-value transactions but infrequent")
            report.append("  → Focus on increasing purchase frequency")
            report.append("  → VIP treatment and exclusive offers")
        else:
            report.append(f"  → Average recency: {r:.0f} days")
            report.append(f"  → Average frequency: {f:.0f} purchases")
            report.append(f"  → Average spend: ₹{m:.2f}")
    
    def _add_marketing_recommendations(self, report, segment_name, row, segment_pct):
        """
        Add specific marketing recommendations for a segment
        """
        if 'Champion' in segment_name or '🌟' in segment_name:
            report.append("  1. VIP rewards program and exclusive access")
            report.append("  2. Request reviews and referrals")
            report.append("  3. Early access to new products")
            report.append("  4. Personalized premium offers")
            report.append(f"  💰 Expected ROI: HIGH (Top {segment_pct:.0f}% of customers)")
        
        elif 'At Risk' in segment_name or '⚠️' in segment_name:
            report.append("  1. URGENT: Win-back campaigns with special offers")
            report.append("  2. Personalized outreach (email/SMS)")
            report.append("  3. Survey to understand dissatisfaction")
            report.append("  4. Limited-time discounts (20-30%)")
            report.append("  ⏰ PRIORITY: HIGH - Act within 2 weeks")
        
        elif 'Hibernating' in segment_name or '😴' in segment_name:
            report.append("  1. Re-engagement email series")
            report.append("  2. 'We miss you' campaigns with incentives")
            report.append("  3. Survey non-responders for data cleaning")
            report.append("  4. Consider cost of retention vs. acquisition")
            report.append("  💡 TIP: Set budget cap - not all can be saved")
        
        elif 'New' in segment_name or '🆕' in segment_name or 'Promising' in segment_name:
            report.append("  1. Onboarding email sequence")
            report.append("  2. Educational content about products")
            report.append("  3. First-purchase follow-up")
            report.append("  4. Encourage second purchase (critical conversion)")
            report.append("  🎯 GOAL: Convert to Loyal/Champions within 90 days")
        
        elif 'Loyal' in segment_name or '💎' in segment_name:
            report.append("  1. Maintain engagement with regular communication")
            report.append("  2. Cross-sell and upsell opportunities")
            report.append("  3. Loyalty points and rewards")
            report.append("  4. Nurture toward Champion status")
            report.append("  📈 OPPORTUNITY: Increase basket size and frequency")
        
        else:
            report.append("  1. Standard marketing campaigns")
            report.append("  2. Monitor segment movement")
            report.append("  3. A/B test messaging and offers")
            report.append("  4. Track engagement metrics")
    
    def _add_business_impact(self, report, df_profiles, total_customers):
        """
        Add business impact analysis
        """
        report.append("\nRevenue Distribution:")
        
        if 'Value_Score' in df_profiles.columns and 'Monetary_mean' in df_profiles.columns:
            # Calculate revenue contribution
            df_profiles['estimated_revenue'] = df_profiles['Recency_count'] * df_profiles['Monetary_mean']
            total_revenue = df_profiles['estimated_revenue'].sum()
            
            for idx, row in df_profiles.iterrows():
                segment_name = row.get('Segment_Name', f'Segment {idx}')
                revenue_contribution = (row['estimated_revenue'] / total_revenue) * 100
                customer_pct = (row['Recency_count'] / total_customers) * 100
                
                efficiency = revenue_contribution / customer_pct if customer_pct > 0 else 0
                
                report.append(f"\n  {segment_name}:")
                report.append(f"    • Revenue Contribution: {revenue_contribution:.1f}%")
                report.append(f"    • Customer Share: {customer_pct:.1f}%")
                report.append(f"    • Efficiency Ratio: {efficiency:.2f}x")
                
                if efficiency > 1.5:
                    report.append(f"    → HIGH EFFICIENCY: Generating above-average revenue per customer")
                elif efficiency > 0.8:
                    report.append(f"    → BALANCED: Revenue aligned with customer share")
                else:
                    report.append(f"    → LOW EFFICIENCY: Underperforming relative to size")
        
        report.append("\n💡 Key Insight:")
        
        # Pareto analysis
        high_value_segments = df_profiles[df_profiles.get('Value_Score', 0) > 60] if 'Value_Score' in df_profiles.columns else df_profiles.head(2)
        high_value_pct = (high_value_segments['Recency_count'].sum() / total_customers) * 100
        
        report.append(f"  • Top-tier customers represent {high_value_pct:.0f}% of your base")
        report.append(f"  • Focus 60% of marketing budget on these segments")
        report.append(f"  • Allocate 30% to growth segments (New/Promising)")
        report.append(f"  • Reserve 10% for retention/win-back campaigns")
    
    def _add_transfer_learning_assessment(self, report, summary_row, sil_score, db_score):
        """
        Add transfer learning assessment
        """
        transferability = summary_row.get('transferability', 'Unknown')
        
        report.append(f"\nTransferability Category: {transferability}")
        
        if sil_score > 0.4 and db_score < 1.5:
            transfer_potential = "HIGH"
            confidence = "90-95%"
            recommendation = "RECOMMENDED: Model can be directly applied to similar domains"
        elif sil_score > 0.25 and db_score < 2.0:
            transfer_potential = "MODERATE"
            confidence = "70-80%"
            recommendation = "CONDITIONAL: Fine-tuning recommended with target domain data"
        else:
            transfer_potential = "LOW"
            confidence = "40-60%"
            recommendation = "NOT RECOMMENDED: Train fresh model on target domain"
        
        report.append(f"Transfer Learning Potential: {transfer_potential}")
        report.append(f"Expected Success Rate: {confidence}")
        report.append(f"\nRecommendation: {recommendation}")
        
        report.append(f"\n{'─'*80}")
        report.append("Transfer Learning Strategy:")
        report.append(f"{'─'*80}")
        
        if transfer_potential == "HIGH":
            report.append("\n1. Direct Application:")
            report.append("   → Apply model to target domain without modification")
            report.append("   → Monitor performance for 2-4 weeks")
            report.append("   → Validate segment assignments manually (sample 100 customers)")
            report.append("\n2. Expected Results:")
            report.append("   → Segment separation should remain strong")
            report.append("   → RFM thresholds transfer well")
            report.append("   → Minimal degradation in quality metrics")
        
        elif transfer_potential == "MODERATE":
            report.append("\n1. Fine-Tuning Approach:")
            report.append("   → Start with source model as initialization")
            report.append("   → Collect 500-1000 target domain samples")
            report.append("   → Re-train with mixed source + target data")
            report.append("   → Adjust RFM thresholds for target domain")
            report.append("\n2. Expected Challenges:")
            report.append("   → Different spending patterns may require threshold adjustment")
            report.append("   → Purchase frequency norms may differ")
            report.append("   → Monitor for segment drift")
        
        else:
            report.append("\n1. Fresh Model Required:")
            report.append("   → Source and target domains are too dissimilar")
            report.append("   → Collect 2000+ target domain samples")
            report.append("   → Train new model from scratch")
            report.append("   → Use source model insights for feature engineering")
            report.append("\n2. Why Transfer Fails:")
            report.append("   → Different customer bases (demographics, behavior)")
            report.append("   → Different price points (affects Monetary metric)")
            report.append("   → Different purchase cycles (affects Recency/Frequency)")
    
    def _add_risks_and_limitations(self, report, summary_row, df_profiles):
        """
        Add risks and limitations section
        """
        balance_ratio = summary_row.get('balance_ratio', 0)
        sil_score = summary_row['silhouette_score']
        
        report.append("\nIdentified Risks:")
        
        risk_count = 0
        
        # Imbalanced segments
        if balance_ratio > 4.0:
            risk_count += 1
            report.append(f"\n  {risk_count}. SEGMENT IMBALANCE (Balance Ratio: {balance_ratio:.2f})")
            report.append("     → Some segments are disproportionately large/small")
            report.append("     → May indicate over-segmentation or under-segmentation")
            report.append("     → Consider: Merging small segments or splitting large ones")
        
        # Low quality
        if sil_score < 0.3:
            risk_count += 1
            report.append(f"\n  {risk_count}. LOW SEGMENTATION QUALITY")
            report.append("     → Segments overlap significantly")
            report.append("     → Customer assignments may be unreliable")
            report.append("     → Risk of misallocated marketing spend")
        
        # Too few segments
        if summary_row['optimal_k'] <= 3:
            risk_count += 1
            report.append(f"\n  {risk_count}. LIMITED SEGMENTATION GRANULARITY")
            report.append(f"     → Only {int(summary_row['optimal_k'])} segments identified")
            report.append("     → May miss important customer sub-groups")
            report.append("     → Consider: Testing k=4 to k=6 for finer segmentation")
        
        # Data limitations
        report.append(f"\n  {risk_count + 1}. DATA LIMITATIONS")
        report.append("     → RFM captures only transactional behavior")
        report.append("     → Missing: Demographics, preferences, channel data")
        report.append("     → Recommendation: Enrich with additional data sources")
        
        if risk_count == 0:
            report.append("\n  ✓ No major risks identified")
            report.append("  → Segmentation quality is acceptable")
            report.append("  → Proceed with confidence")
    
    def _add_action_items(self, report, summary_row, sil_score, db_score):
        """
        Add recommended action items
        """
        report.append("\nImmediate Actions (This Week):")
        report.append("  1. Review segment profiles with marketing team")
        report.append("  2. Validate segment assignments with sample customers")
        report.append("  3. Define segment-specific KPIs")
        report.append("  4. Set up tracking and monitoring dashboards")
        
        report.append("\nShort-term Actions (This Month):")
        report.append("  1. Design segment-specific marketing campaigns")
        report.append("  2. Create personalized messaging for each segment")
        report.append("  3. Set campaign budgets based on segment value")
        report.append("  4. Launch pilot campaigns (20% of budget)")
        
        report.append("\nMedium-term Actions (Next Quarter):")
        report.append("  1. Analyze campaign performance by segment")
        report.append("  2. Monitor segment migration (who moves between segments)")
        report.append("  3. Refine segmentation based on outcomes")
        report.append("  4. Expand successful campaigns to full budget")
        
        report.append("\nLong-term Actions (Next 6 Months):")
        report.append("  1. Integrate segmentation into CRM systems")
        report.append("  2. Automate segment-based triggers")
        report.append("  3. Develop predictive models for segment transitions")
        report.append("  4. Calculate actual ROI by segment")
        
        if sil_score < 0.3:
            report.append("\n⚠️  PRIORITY ACTION:")
            report.append("  → Segmentation quality is below threshold")
            report.append("  → Schedule review meeting to decide:")
            report.append("     a) Collect more/better data")
            report.append("     b) Try alternative clustering methods")
            report.append("     c) Add non-RFM features")
    
    def _create_domain_dashboard(self, domain_id, summary_row, df_profiles, reports_dir):
        """
        Create visual dashboard for domain
        """
        try:
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle(f"Domain Analysis Dashboard: {summary_row['domain_name']}", 
                         fontsize=18, fontweight='bold', y=0.98)
            
            # Create grid
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Quality Metrics (Top Left)
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_quality_gauges(ax1, summary_row)
            
            # 2. Segment Distribution (Top Middle)
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_segment_pie(ax2, df_profiles)
            
            # 3. Value Scores (Top Right)
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_value_bars(ax3, df_profiles)
            
            # 4. RFM Heatmap (Middle Left)
            ax4 = fig.add_subplot(gs[1, 0])
            self._plot_rfm_heatmap(ax4, df_profiles)
            
            # 5. Revenue Contribution (Middle)
            ax5 = fig.add_subplot(gs[1, 1:])
            self._plot_revenue_contribution(ax5, df_profiles)
            
            # 6. Segment Comparison (Bottom)
            ax6 = fig.add_subplot(gs[2, :])
            self._plot_segment_comparison(ax6, df_profiles)
            
            # Save
            dashboard_file = f'{reports_dir}/{domain_id}_dashboard.png'
            plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved dashboard: {dashboard_file}")
        except Exception as e:
            print(f"⚠️  Error creating dashboard for {domain_id}: {str(e)}")
            plt.close('all')
    
    def _plot_quality_gauges(self, ax, summary_row):
        """Plot quality metrics as gauge"""
        sil = summary_row['silhouette_score']
        db = summary_row['davies_bouldin_index']
        
        # Normalize DB score (invert and cap at 3)
        db_norm = max(0, min(1, 1 - (db / 3)))
        
        metrics = ['Silhouette\nScore', 'Davies-Bouldin\n(inverted)']
        values = [sil, db_norm]
        colors = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in values]
        
        bars = ax.barh(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_title('Clustering Quality', fontweight='bold', fontsize=12)
        ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Good')
        ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, label='Acceptable')
        
        # Add value labels
        for bar, val in zip(bars, [sil, db]):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontweight='bold')
        
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_segment_pie(self, ax, df_profiles):
        """Plot segment distribution pie chart"""
        sizes = df_profiles['Recency_count']
        labels = [name.split(' ', 1)[1] if ' ' in name else name 
                 for name in df_profiles.get('Segment_Name', df_profiles.index)]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        ax.set_title('Customer Distribution', fontweight='bold', fontsize=12)
    
    def _plot_value_bars(self, ax, df_profiles):
        """Plot value scores as horizontal bars"""
        if 'Value_Score' not in df_profiles.columns:
            ax.text(0.5, 0.5, 'Value scores not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Customer Value Scores', fontweight='bold', fontsize=12)
            return
        
        labels = [name.split(' ', 1)[1] if ' ' in name else name 
                 for name in df_profiles.get('Segment_Name', df_profiles.index)]
        values = df_profiles['Value_Score']
        
        colors = ['green' if v > 70 else 'orange' if v > 50 else 'red' for v in values]
        
        bars = ax.barh(labels, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Value Score (0-100)', fontweight='bold')
        ax.set_title('Customer Value Scores', fontweight='bold', fontsize=12)
        ax.axvline(x=70, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=50, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                   f'{val:.0f}', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
    
    def _plot_rfm_heatmap(self, ax, df_profiles):
        """Plot RFM characteristics as heatmap"""
        rfm_data = df_profiles[['Recency_mean', 'Frequency_mean', 'Monetary_mean']].copy()
        
        # Normalize each column to 0-1 scale (invert Recency)
        rfm_data['Recency_mean'] = 1 - (rfm_data['Recency_mean'] / rfm_data['Recency_mean'].max())
        rfm_data['Frequency_mean'] = rfm_data['Frequency_mean'] / rfm_data['Frequency_mean'].max()
        rfm_data['Monetary_mean'] = rfm_data['Monetary_mean'] / rfm_data['Monetary_mean'].max()
        
        labels = [name.split(' ', 1)[1] if ' ' in name else name 
                 for name in df_profiles.get('Segment_Name', df_profiles.index)]
        
        rfm_data.index = labels
        rfm_data.columns = ['Recency\n(inverted)', 'Frequency', 'Monetary']
        
        sns.heatmap(rfm_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   ax=ax, cbar_kws={'label': 'Normalized Score'},
                   linewidths=0.5, linecolor='black', vmin=0, vmax=1)
        
        ax.set_title('RFM Profile Heatmap', fontweight='bold', fontsize=12)
        ax.set_ylabel('')
    
    def _plot_revenue_contribution(self, ax, df_profiles):
        """Plot revenue contribution analysis"""
        if 'Monetary_mean' not in df_profiles.columns:
            ax.text(0.5, 0.5, 'Revenue data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calculate estimated revenue
        df_profiles_copy = df_profiles.copy()
        df_profiles_copy['estimated_revenue'] = df_profiles_copy['Recency_count'] * df_profiles_copy['Monetary_mean']
        total_revenue = df_profiles_copy['estimated_revenue'].sum()
        total_customers = df_profiles_copy['Recency_count'].sum()
        
        df_profiles_copy['revenue_pct'] = (df_profiles_copy['estimated_revenue'] / total_revenue) * 100
        df_profiles_copy['customer_pct'] = (df_profiles_copy['Recency_count'] / total_customers) * 100
        
        labels = [name.split(' ', 1)[1] if ' ' in name else name 
                 for name in df_profiles_copy.get('Segment_Name', df_profiles_copy.index)]
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df_profiles_copy['customer_pct'], width, 
                       label='Customer %', alpha=0.8, color='skyblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, df_profiles_copy['revenue_pct'], width, 
                       label='Revenue %', alpha=0.8, color='lightgreen', edgecolor='black')
        
        ax.set_xlabel('Segment', fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title('Revenue vs Customer Distribution', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add efficiency indicators
        for i, (c_pct, r_pct) in enumerate(zip(df_profiles_copy['customer_pct'], df_profiles_copy['revenue_pct'])):
            efficiency = r_pct / c_pct if c_pct > 0 else 0
            if efficiency > 1.2:
                marker = '★'
            elif efficiency < 0.8:
                marker = '▼'
            else:
                marker = ''
            
            if marker:
                ax.text(i, max(c_pct, r_pct) + 2, marker, 
                       ha='center', fontsize=14, fontweight='bold')
    
    def _plot_segment_comparison(self, ax, df_profiles):
        """Plot comprehensive segment comparison"""
        labels = [name.split(' ', 1)[1] if ' ' in name else name 
                 for name in df_profiles.get('Segment_Name', df_profiles.index)]
        
        x = np.arange(len(labels))
        
        # Normalize metrics for comparison
        recency_norm = 1 - (df_profiles['Recency_mean'] / df_profiles['Recency_mean'].max())
        frequency_norm = df_profiles['Frequency_mean'] / df_profiles['Frequency_mean'].max()
        monetary_norm = df_profiles['Monetary_mean'] / df_profiles['Monetary_mean'].max()
        
        ax.plot(x, recency_norm, 'o-', linewidth=2, markersize=8, label='Recency (inverted)', color='coral')
        ax.plot(x, frequency_norm, 's-', linewidth=2, markersize=8, label='Frequency', color='skyblue')
        ax.plot(x, monetary_norm, '^-', linewidth=2, markersize=8, label='Monetary', color='lightgreen')
        
        ax.set_xlabel('Segment', fontweight='bold', fontsize=11)
        ax.set_ylabel('Normalized Score (0-1)', fontweight='bold', fontsize=11)
        ax.set_title('Segment RFM Profile Comparison', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)
        
        # Add reference line
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    def generate_comparative_summary(self, df_summary, reports_dir):
        """
        Generate cross-domain comparative summary
        """
        print(f"\n{'='*80}")
        print("📊 Generating Cross-Domain Comparative Summary")
        print(f"{'='*80}")
        
        try:
            # Create comparative visualizations
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle('Cross-Domain Comparative Analysis', fontsize=18, fontweight='bold')
            
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # 1. Quality comparison
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_quality_comparison(ax1, df_summary)
            
            # 2. Transferability analysis
            ax2 = fig.add_subplot(gs[1, 0])
            self._plot_transferability_analysis(ax2, df_summary)
            
            # 3. Segment distribution
            ax3 = fig.add_subplot(gs[1, 1])
            self._plot_segment_distribution_comparison(ax3, df_summary)
            
            # 4. Customer volume
            ax4 = fig.add_subplot(gs[2, 0])
            self._plot_customer_volume(ax4, df_summary)
            
            # 5. Balance analysis
            ax5 = fig.add_subplot(gs[2, 1])
            self._plot_balance_analysis(ax5, df_summary)
            
            # Save
            summary_file = f'{reports_dir}/cross_domain_comparative_summary.png'
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved: {summary_file}")
        except Exception as e:
            print(f"⚠️  Error creating comparative summary: {str(e)}")
            plt.close('all')
        
        # Create text summary
        self._create_comparative_text_report(df_summary, reports_dir)
    
    def _plot_quality_comparison(self, ax, df_summary):
        """Plot quality metrics comparison across domains"""
        domains = [name.split('→')[0].strip() for name in df_summary['domain_name']]
        x = np.arange(len(domains))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df_summary['silhouette_score'], width,
                      label='Silhouette Score', alpha=0.8, color='steelblue', edgecolor='black')
        
        # Normalize Davies-Bouldin (invert and scale)
        db_norm = 1 - (df_summary['davies_bouldin_index'] / 3)
        db_norm = db_norm.clip(0, 1)
        bars2 = ax.bar(x + width/2, db_norm, width,
                      label='Davies-Bouldin (inv)', alpha=0.8, color='coral', edgecolor='black')
        
        ax.set_xlabel('Domain', fontweight='bold', fontsize=11)
        ax.set_ylabel('Score', fontweight='bold', fontsize=11)
        ax.set_title('Clustering Quality Comparison Across Domains', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add quality threshold lines
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=0.35, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    def _plot_transferability_analysis(self, ax, df_summary):
        """Plot transferability categories"""
        if 'transferability' not in df_summary.columns:
            ax.text(0.5, 0.5, 'Transferability data not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Transfer Learning Potential', fontweight='bold', fontsize=12)
            return
        
        trans_counts = df_summary['transferability'].value_counts()
        colors_map = {'No Finetune': 'green', 'Partial': 'orange', 
                 'Partial (Low)': 'yellow', 'New Model': 'red'}
        
        colors = [colors_map.get(x, 'gray') for x in trans_counts.index]
        
        wedges, texts, autotexts = ax.pie(trans_counts.values, 
                                          labels=trans_counts.index,
                                          autopct='%1.0f%%',
                                          colors=colors,
                                          startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        ax.set_title('Transfer Learning Potential', fontweight='bold', fontsize=12)
    
    def _plot_segment_distribution_comparison(self, ax, df_summary):
        """Plot optimal k distribution"""
        domains = [name.split('→')[0].strip() for name in df_summary['domain_name']]
        
        bars = ax.bar(domains, df_summary['optimal_k'], alpha=0.8, 
                     color='lightgreen', edgecolor='black')
        
        ax.set_xlabel('Domain', fontweight='bold', fontsize=11)
        ax.set_ylabel('Number of Segments', fontweight='bold', fontsize=11)
        ax.set_title('Optimal Segment Count by Domain', fontweight='bold', fontsize=12)
        ax.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
    
    def _plot_customer_volume(self, ax, df_summary):
        """Plot customer volume by domain"""
        domains = [name.split('→')[0].strip() for name in df_summary['domain_name']]
        
        bars = ax.barh(domains, df_summary['n_customers'], alpha=0.8,
                      color='skyblue', edgecolor='black')
        
        ax.set_xlabel('Number of Customers', fontweight='bold', fontsize=11)
        ax.set_title('Customer Base Size by Domain', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 20, bar.get_y() + bar.get_height()/2,
                   f'{int(width):,}',
                   va='center', fontweight='bold')
    
    def _plot_balance_analysis(self, ax, df_summary):
        """Plot segment balance ratios"""
        domains = [name.split('→')[0].strip() for name in df_summary['domain_name']]
        ratios = df_summary.get('balance_ratio', [1] * len(domains))
        
        colors = ['green' if r < 2 else 'orange' if r < 4 else 'red' for r in ratios]
        
        bars = ax.barh(domains, ratios, alpha=0.8, color=colors, edgecolor='black')
        
        ax.set_xlabel('Balance Ratio', fontweight='bold', fontsize=11)
        ax.set_title('Segment Balance Analysis', fontweight='bold', fontsize=12)
        ax.axvline(x=2, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Well-balanced')
        ax.axvline(x=4, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Acceptable')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}',
                   va='center', fontweight='bold')
    
    def _create_comparative_text_report(self, df_summary, reports_dir):
        """Create text-based comparative report"""
        report = []
        report.append("="*80)
        report.append("CROSS-DOMAIN COMPARATIVE ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        
        report.append("\n" + "="*80)
        report.append("📊 EXECUTIVE SUMMARY")
        report.append("="*80)
        
        report.append(f"\nTotal Domains Analyzed: {len(df_summary)}")
        report.append(f"Total Customers: {df_summary['n_customers'].sum():,}")
        report.append(f"Average Silhouette Score: {df_summary['silhouette_score'].mean():.3f}")
        
        # Best and worst performers
        best = df_summary.loc[df_summary['silhouette_score'].idxmax()]
        worst = df_summary.loc[df_summary['silhouette_score'].idxmin()]
        
        report.append(f"\n🏆 Best Performing Domain:")
        report.append(f"   {best['domain_name']}")
        report.append(f"   Silhouette Score: {best['silhouette_score']:.3f}")
        report.append(f"   Quality: {'EXCELLENT' if best['silhouette_score'] > 0.5 else 'GOOD'}")
        
        report.append(f"\n⚠️  Weakest Performing Domain:")
        report.append(f"   {worst['domain_name']}")
        report.append(f"   Silhouette Score: {worst['silhouette_score']:.3f}")
        report.append(f"   Recommendation: {'Acceptable' if worst['silhouette_score'] > 0.25 else 'Needs improvement'}")
        
        # Transfer learning summary
        if 'transferability' in df_summary.columns:
            report.append("\n" + "="*80)
            report.append("🔄 TRANSFER LEARNING READINESS")
            report.append("="*80)
            
            for trans_type in df_summary['transferability'].unique():
                subset = df_summary[df_summary['transferability'] == trans_type]
                report.append(f"\n{trans_type}:")
                report.append(f"  Domains: {len(subset)}")
                report.append(f"  Avg Silhouette: {subset['silhouette_score'].mean():.3f}")
        
        # Recommendations
        report.append("\n" + "="*80)
        report.append("💡 KEY RECOMMENDATIONS")
        report.append("="*80)
        
        excellent_domains = df_summary[df_summary['silhouette_score'] > 0.4]
        poor_domains = df_summary[df_summary['silhouette_score'] < 0.25]
        
        report.append(f"\n1. DEPLOYMENT READY ({len(excellent_domains)} domains):")
        for _, row in excellent_domains.iterrows():
            report.append(f"   ✓ {row['domain_name']}")
        
        if len(poor_domains) > 0:
            report.append(f"\n2. REQUIRES IMPROVEMENT ({len(poor_domains)} domains):")
            for _, row in poor_domains.iterrows():
                report.append(f"   ✗ {row['domain_name']}")
        
        report.append("\n3. NEXT STEPS:")
        report.append("   → Prioritize high-quality domains for pilot campaigns")
        report.append("   → Test transfer learning on 'No Finetune' categories")
        report.append("   → Improve data collection for weak performers")
        
        # Save report
        report_file = f'{reports_dir}/cross_domain_summary_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✓ Saved: {report_file}")


def main():
    """
    Main function to generate all reports
    """
    print("\n" + "📊"*40)
    print("DOMAIN-WISE RFM ANALYSIS REPORT GENERATOR")
    print("📊"*40)
    
    generator = DomainReportGenerator()
    generator.generate_all_reports()
    
    print("\n" + "🎉"*40)
    print("REPORT GENERATION COMPLETE!")
    print("🎉"*40)
    
    print("\n📁 Generated Reports:")
    print("   • Individual domain analysis reports (TXT)")
    print("   • Individual domain dashboards (PNG)")
    print("   • Cross-domain comparative summary (PNG)")
    print("   • Cross-domain summary report (TXT)")
    
    print("\n✅ All reports saved in 'reports/' directory")


if __name__ == "__main__":
    main()