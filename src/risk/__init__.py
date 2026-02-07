"""
Risk Stratification Module

Converts ML risk scores into clinical risk categories (Green/Yellow/Red)
with rule-based overrides and personalized recommendations.
"""

from src.risk.stratification import RiskStratifier, stratify_risk
from src.risk.rules import apply_override_rules, check_override_rules

__all__ = [
    'RiskStratifier',
    'stratify_risk',
    'apply_override_rules',
    'check_override_rules'
]
