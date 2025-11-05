"""
Transfer Learning Framework for Customer Segmentation
======================================================

A complete framework for assessing transferability and executing transfer
learning between customer domains using RFM (Recency, Frequency, Monetary) features.

Author: Member 3 (Research Lead)
Date: Week 3-4, 2024

Modules:
--------
- metrics: Research-backed transferability metrics (MMD, JS, KS, Wasserstein, Correlation)
- decision_engine: Strategy recommendation logic
- framework: Main framework class for end-to-end workflows
- calibration: Threshold calibration using experimental data
- validation: Framework accuracy validation

Quick Start:
------------
>>> from framework import TransferLearningFramework
>>> fw = TransferLearningFramework()
>>> fw.load_data('source_RFM.csv', 'target_RFM.csv')
>>> fw.calculate_transferability()
>>> rec = fw.recommend_strategy()
>>> print(f"Recommendation: {rec.transferability_level.value}")

Or use the quick assessment function:
>>> from framework import quick_transfer_assessment
>>> rec = quick_transfer_assessment('source.csv', 'target.csv', 'My Pair')

For comprehensive demos:
>>> python demo_framework.py

For complete documentation, see README.md
"""

__version__ = '1.0.0'
__author__ = 'Member 3 (Research Lead)'

# Import main classes for easy access
from .framework import TransferLearningFramework, quick_transfer_assessment
from .metrics import TransferabilityMetrics, quick_transferability_check
from .decision_engine import (
    DecisionEngine, 
    TransferRecommendation, 
    TransferStrategy, 
    TransferabilityLevel
)

# Define what gets imported with "from week3 import *"
__all__ = [
    # Framework
    'TransferLearningFramework',
    'quick_transfer_assessment',
    
    # Metrics
    'TransferabilityMetrics',
    'quick_transferability_check',
    
    # Decision Engine
    'DecisionEngine',
    'TransferRecommendation',
    'TransferStrategy',
    'TransferabilityLevel',
]

# Package metadata
__title__ = 'Transfer Learning Framework'
__description__ = 'Transfer learning framework for customer segmentation'
__url__ = 'https://github.com/akashmadisetty/CDAT'
__license__ = 'MIT'
