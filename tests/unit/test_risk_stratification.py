"""
Unit tests for risk stratification module.
"""

import pytest

from src.risk.stratification import RiskStratifier, stratify_risk
from src.risk.rules import evaluate_condition, check_override_rules


class TestRiskStratifier:
    """Tests for RiskStratifier class."""

    def test_threshold_based_stratification_green(self):
        """Test Green risk stratification."""
        stratifier = RiskStratifier()

        result = stratifier.stratify(
            ml_score=0.25,
            patient_id=1
        )

        assert result['risk_level'] == 'Green'
        assert result['ml_score'] == 0.25
        assert not result['override_applied']

    def test_threshold_based_stratification_yellow(self):
        """Test Yellow risk stratification."""
        stratifier = RiskStratifier()

        result = stratifier.stratify(
            ml_score=0.50,
            patient_id=1
        )

        assert result['risk_level'] == 'Yellow'
        assert not result['override_applied']

    def test_threshold_based_stratification_red(self):
        """Test Red risk stratification."""
        stratifier = RiskStratifier()

        result = stratifier.stratify(
            ml_score=0.75,
            patient_id=1
        )

        assert result['risk_level'] == 'Red'
        assert not result['override_applied']

    def test_override_tachycardia(self):
        """Test override for high resting heart rate."""
        stratifier = RiskStratifier()

        result = stratifier.stratify(
            ml_score=0.25,  # Would be Green
            features={'resting_hr_estimate': 105},  # Triggers override
            patient_id=1
        )

        assert result['risk_level'] == 'Red'
        assert result['override_applied']
        assert result['override_reason'] == 'tachycardia_pattern'

    def test_override_extreme_sedentary(self):
        """Test override for extremely low activity."""
        stratifier = RiskStratifier()

        result = stratifier.stratify(
            ml_score=0.30,  # Would be Yellow
            features={'steps_avg_30d': 1500},  # Very low
            patient_id=1
        )

        assert result['risk_level'] == 'Red'
        assert result['override_applied']
        assert result['override_reason'] == 'extreme_sedentary'

    def test_recommendations_included(self):
        """Test that recommendations are included."""
        stratifier = RiskStratifier()

        result = stratifier.stratify(
            ml_score=0.75,
            patient_id=1
        )

        assert 'recommendations' in result
        assert len(result['recommendations']) > 0

    def test_risk_metadata_included(self):
        """Test that risk metadata is included."""
        stratifier = RiskStratifier()

        result = stratifier.stratify(
            ml_score=0.75,
            patient_id=1
        )

        assert 'risk_metadata' in result
        assert 'color' in result['risk_metadata']
        assert 'icon' in result['risk_metadata']


class TestRuleEvaluation:
    """Tests for rule evaluation."""

    def test_evaluate_greater_than(self):
        """Test > operator."""
        features = {'resting_hr_estimate': 95}
        result = evaluate_condition('resting_hr_estimate > 90', features)
        assert result == True

    def test_evaluate_less_than(self):
        """Test < operator."""
        features = {'sleep_hours_avg': 5.5}
        result = evaluate_condition('sleep_hours_avg < 6', features)
        assert result == True

    def test_evaluate_false_condition(self):
        """Test condition that is false."""
        features = {'resting_hr_estimate': 70}
        result = evaluate_condition('resting_hr_estimate > 90', features)
        assert result == False

    def test_evaluate_missing_feature(self):
        """Test evaluation with missing feature."""
        features = {'other_feature': 100}
        result = evaluate_condition('resting_hr_estimate > 90', features)
        assert result == False

    def test_check_override_rules(self):
        """Test full override rule check."""
        features = {'resting_hr_estimate': 105}
        override_rules = {
            'tachycardia_pattern': {
                'condition': 'resting_hr_estimate > 100',
                'force_level': 'red',
                'reason': 'Elevated resting heart rate'
            }
        }

        result = check_override_rules(features, override_rules)

        assert result['override_triggered'] == True
        assert result['triggered_rule'] == 'tachycardia_pattern'
        assert result['force_level'] == 'Red'


class TestStratifyRiskFunction:
    """Tests for convenience function."""

    def test_stratify_risk_function(self):
        """Test stratify_risk convenience function."""
        result = stratify_risk(
            ml_score=0.50,
            patient_id=123
        )

        assert 'risk_level' in result
        assert 'ml_score' in result
        assert result['patient_id'] == 123
