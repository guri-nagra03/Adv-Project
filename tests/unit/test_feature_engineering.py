"""
Unit tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.feature_engineering import (
    estimate_resting_hr,
    calculate_activity_score,
    calculate_rolling_average,
    calculate_sedentary_ratio,
    calculate_workout_consistency,
    calculate_hr_variability,
    calculate_mood_stress_ratio,
    create_cardiovascular_features,
    get_feature_columns
)


class TestRestingHREstimation:
    """Tests for resting heart rate estimation."""

    def test_estimate_resting_hr_low_activity(self, sample_raw_data):
        """Test resting HR estimation from low activity periods."""
        df = sample_raw_data.copy()
        # Set some low activity days
        df.loc[0:5, 'steps'] = 500
        df.loc[0:5, 'heart_rate_avg'] = 65

        resting_hr = estimate_resting_hr(df)

        assert resting_hr is not None
        assert 60 <= resting_hr <= 100

    def test_estimate_resting_hr_no_low_activity(self, sample_raw_data):
        """Test when no low activity periods exist."""
        df = sample_raw_data.copy()
        # All high activity
        df['steps'] = 15000

        resting_hr = estimate_resting_hr(df)

        # Should return overall median as fallback
        assert resting_hr is not None


class TestActivityScore:
    """Tests for activity score calculation."""

    def test_calculate_activity_score(self):
        """Test activity score calculation."""
        score = calculate_activity_score(
            steps=10000,
            active_minutes=60,
            distance_km=8.0
        )

        assert score > 0
        assert score <= 100

    def test_activity_score_zero_values(self):
        """Test activity score with zero values."""
        score = calculate_activity_score(
            steps=0,
            active_minutes=0,
            distance_km=0
        )

        assert score == 0

    def test_activity_score_high_values(self):
        """Test activity score with high values."""
        score = calculate_activity_score(
            steps=20000,
            active_minutes=120,
            distance_km=15.0
        )

        assert score > 50  # Should be high


class TestRollingAverage:
    """Tests for rolling average calculation."""

    def test_rolling_average_7_days(self, sample_raw_data):
        """Test 7-day rolling average."""
        df = sample_raw_data.copy()
        df = df.sort_values(['user_id', 'date'])

        df['sleep_avg_7d'] = calculate_rolling_average(
            df.groupby('user_id')['sleep_hours'],
            window=7
        )

        assert df['sleep_avg_7d'].notna().any()
        # First few values should be NaN
        assert pd.isna(df['sleep_avg_7d'].iloc[0])

    def test_rolling_average_30_days(self, sample_raw_data):
        """Test 30-day rolling average."""
        df = sample_raw_data.copy()
        df = df.sort_values(['user_id', 'date'])

        df['steps_avg_30d'] = calculate_rolling_average(
            df.groupby('user_id')['steps'],
            window=30
        )

        assert df['steps_avg_30d'].notna().any()


class TestSedentaryRatio:
    """Tests for sedentary ratio calculation."""

    def test_sedentary_ratio_all_sedentary(self):
        """Test when all days are sedentary."""
        steps = pd.Series([3000, 4000, 2000, 3500])

        ratio = calculate_sedentary_ratio(steps, threshold=5000, window=4)

        assert ratio.iloc[-1] == 1.0

    def test_sedentary_ratio_no_sedentary(self):
        """Test when no days are sedentary."""
        steps = pd.Series([10000, 12000, 8000, 9000])

        ratio = calculate_sedentary_ratio(steps, threshold=5000, window=4)

        assert ratio.iloc[-1] == 0.0


class TestWorkoutConsistency:
    """Tests for workout consistency calculation."""

    def test_workout_consistency_all_workouts(self):
        """Test when all days have workouts."""
        workouts = pd.Series(['Running', 'Cycling', 'Walking', 'Running'])

        consistency = calculate_workout_consistency(workouts, window=4)

        assert consistency.iloc[-1] == 1.0

    def test_workout_consistency_no_workouts(self):
        """Test when no days have workouts."""
        workouts = pd.Series([None, None, None, None])

        consistency = calculate_workout_consistency(workouts, window=4)

        assert consistency.iloc[-1] == 0.0


class TestHRVariability:
    """Tests for heart rate variability calculation."""

    def test_hr_variability(self):
        """Test HR variability calculation."""
        hr = pd.Series([70, 72, 68, 75, 71, 69, 73])

        variability = calculate_hr_variability(hr, window=7)

        assert variability.iloc[-1] > 0
        assert variability.iloc[-1] < 20  # Reasonable range


class TestMoodStressRatio:
    """Tests for mood stress ratio calculation."""

    def test_mood_stress_ratio_all_stressed(self):
        """Test when all moods are stressed."""
        mood = pd.Series(['Stressed'] * 14)

        ratio = calculate_mood_stress_ratio(mood, window=14)

        assert ratio.iloc[-1] == 1.0

    def test_mood_stress_ratio_no_stress(self):
        """Test when no stressed moods."""
        mood = pd.Series(['Happy'] * 14)

        ratio = calculate_mood_stress_ratio(mood, window=14)

        assert ratio.iloc[-1] == 0.0


class TestFeatureCreation:
    """Tests for complete feature creation pipeline."""

    def test_create_cardiovascular_features(self, sample_raw_data):
        """Test full feature engineering pipeline."""
        features_df = create_cardiovascular_features(sample_raw_data)

        assert len(features_df) > 0
        assert 'user_id' in features_df.columns
        assert 'date' in features_df.columns

        # Check feature columns exist
        feature_cols = get_feature_columns()
        for col in feature_cols:
            assert col in features_df.columns

    def test_get_feature_columns(self):
        """Test feature column list."""
        feature_cols = get_feature_columns()

        assert len(feature_cols) == 7
        assert 'resting_hr_estimate' in feature_cols
        assert 'activity_score' in feature_cols
        assert 'sleep_hours_avg' in feature_cols

    def test_features_within_reasonable_range(self, sample_raw_data):
        """Test that features are within reasonable ranges."""
        features_df = create_cardiovascular_features(sample_raw_data)

        # Remove NaN rows for this test
        features_df = features_df.dropna()

        if len(features_df) > 0:
            # Resting HR should be reasonable
            assert features_df['resting_hr_estimate'].between(40, 120).all()

            # Activity score should be 0-100
            assert features_df['activity_score'].between(0, 100).all()

            # Ratios should be 0-1
            assert features_df['sedentary_ratio'].between(0, 1).all()
            assert features_df['workout_consistency'].between(0, 1).all()
            assert features_df['mood_stress_ratio'].between(0, 1).all()
