"""
Unit tests for synthetic label generation.
"""

import pytest
import pandas as pd
import numpy as np

from src.ml.label_generator import (
    generate_synthetic_labels,
    evaluate_high_risk_conditions,
    evaluate_medium_risk_conditions
)


class TestLabelGeneration:
    """Tests for synthetic label generation."""

    def test_generate_labels_basic(self, sample_features):
        """Test basic label generation."""
        labels = generate_synthetic_labels(sample_features)

        assert len(labels) == len(sample_features)
        assert labels.dtype == np.int64
        assert labels.isin([0, 1, 2]).all()

    def test_high_risk_elevated_hr(self):
        """Test high risk label for elevated resting HR."""
        features = pd.DataFrame({
            'resting_hr_estimate': [95],  # > 90
            'sleep_hours_avg': [7.5],
            'activity_score': [60],
            'sedentary_ratio': [0.3]
        })

        labels = generate_synthetic_labels(features)

        assert labels.iloc[0] == 2  # High risk

    def test_high_risk_low_sleep(self):
        """Test high risk label for insufficient sleep."""
        features = pd.DataFrame({
            'resting_hr_estimate': [70],
            'sleep_hours_avg': [5.5],  # < 6
            'activity_score': [60],
            'sedentary_ratio': [0.3]
        })

        labels = generate_synthetic_labels(features)

        assert labels.iloc[0] == 2  # High risk

    def test_medium_risk_borderline_hr(self):
        """Test medium risk label for borderline HR."""
        features = pd.DataFrame({
            'resting_hr_estimate': [85],  # 75-90 range
            'sleep_hours_avg': [7.5],
            'activity_score': [60],
            'sedentary_ratio': [0.3]
        })

        labels = generate_synthetic_labels(features)

        # Should be medium or high (depending on other factors)
        assert labels.iloc[0] in [1, 2]

    def test_low_risk_healthy_values(self):
        """Test low risk label for healthy values."""
        features = pd.DataFrame({
            'resting_hr_estimate': [70],  # Healthy
            'sleep_hours_avg': [7.5],  # Healthy
            'activity_score': [70],  # High activity
            'sedentary_ratio': [0.2]  # Low sedentary
        })

        labels = generate_synthetic_labels(features)

        assert labels.iloc[0] == 0  # Low risk

    def test_label_distribution(self, sample_features):
        """Test that label distribution is reasonable."""
        # Create larger dataset with varied features
        np.random.seed(42)
        n = 1000

        features = pd.DataFrame({
            'resting_hr_estimate': np.random.normal(75, 10, n),
            'sleep_hours_avg': np.random.normal(7, 1, n),
            'activity_score': np.random.normal(60, 20, n),
            'sedentary_ratio': np.random.uniform(0, 1, n)
        })

        labels = generate_synthetic_labels(features)

        # Check distribution
        label_counts = labels.value_counts()

        # Should have all three classes
        assert 0 in label_counts.index
        assert 1 in label_counts.index
        assert 2 in label_counts.index

        # Proportions should be reasonable (not all same class)
        assert label_counts.min() / label_counts.max() > 0.1


class TestConditionEvaluation:
    """Tests for condition evaluation functions."""

    def test_evaluate_high_risk_conditions(self):
        """Test high risk condition evaluation."""
        features = pd.Series({
            'resting_hr_estimate': 95,
            'sleep_hours_avg': 5.5,
            'activity_score': 20,
            'sedentary_ratio': 0.8
        })

        is_high_risk = evaluate_high_risk_conditions(features)

        assert is_high_risk == True

    def test_evaluate_medium_risk_conditions(self):
        """Test medium risk condition evaluation."""
        features = pd.Series({
            'resting_hr_estimate': 82,
            'sleep_hours_avg': 6.5,
            'activity_score': 45,
            'sedentary_ratio': 0.4
        })

        is_medium_risk = evaluate_medium_risk_conditions(features)

        assert is_medium_risk == True

    def test_no_risk_conditions(self):
        """Test when no risk conditions are met."""
        features = pd.Series({
            'resting_hr_estimate': 68,
            'sleep_hours_avg': 8.0,
            'activity_score': 80,
            'sedentary_ratio': 0.15
        })

        is_high_risk = evaluate_high_risk_conditions(features)
        is_medium_risk = evaluate_medium_risk_conditions(features)

        assert is_high_risk == False
        assert is_medium_risk == False
