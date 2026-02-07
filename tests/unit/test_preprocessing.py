"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessing import (
    clean_data,
    validate_schema,
    handle_missing_values,
    detect_outliers
)


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_valid_schema(self, sample_raw_data):
        """Test that valid schema passes validation."""
        is_valid, errors = validate_schema(sample_raw_data)
        assert is_valid
        assert len(errors) == 0

    def test_missing_required_column(self, sample_raw_data):
        """Test detection of missing required column."""
        df = sample_raw_data.drop(columns=['user_id'])
        is_valid, errors = validate_schema(df)
        assert not is_valid
        assert any('user_id' in err for err in errors)

    def test_invalid_data_type(self, sample_raw_data):
        """Test detection of invalid data types."""
        df = sample_raw_data.copy()
        df['steps'] = 'invalid'  # Should be numeric
        is_valid, errors = validate_schema(df)
        assert not is_valid


class TestMissingValueHandling:
    """Tests for missing value handling."""

    def test_handle_missing_steps(self, sample_raw_data):
        """Test that missing steps are filled with 0."""
        df = sample_raw_data.copy()
        df.loc[0, 'steps'] = np.nan

        df_clean = handle_missing_values(df)

        assert df_clean.loc[0, 'steps'] == 0

    def test_handle_missing_sleep(self, sample_raw_data):
        """Test that missing sleep is filled with median."""
        df = sample_raw_data.copy()
        median_sleep = df['sleep_hours'].median()
        df.loc[0, 'sleep_hours'] = np.nan

        df_clean = handle_missing_values(df)

        assert df_clean.loc[0, 'sleep_hours'] == median_sleep

    def test_forward_fill_heart_rate(self, sample_raw_data):
        """Test forward fill for heart rate."""
        df = sample_raw_data.copy()
        df = df.sort_values(['user_id', 'date'])
        df.loc[1, 'heart_rate_avg'] = np.nan

        df_clean = handle_missing_values(df)

        # Should be filled with previous value
        assert df_clean.loc[1, 'heart_rate_avg'] == df.loc[0, 'heart_rate_avg']


class TestOutlierDetection:
    """Tests for outlier detection."""

    def test_detect_outliers_iqr(self, sample_raw_data):
        """Test IQR-based outlier detection."""
        df = sample_raw_data.copy()
        # Add extreme outlier
        df.loc[0, 'steps'] = 100000

        outlier_mask = detect_outliers(df, 'steps', method='iqr')

        assert outlier_mask[0] == True
        assert outlier_mask.sum() >= 1

    def test_detect_outliers_zscore(self, sample_raw_data):
        """Test Z-score based outlier detection."""
        df = sample_raw_data.copy()
        # Add extreme outlier
        df.loc[0, 'heart_rate_avg'] = 200

        outlier_mask = detect_outliers(df, 'heart_rate_avg', method='zscore')

        assert outlier_mask[0] == True


class TestCleanData:
    """Tests for complete data cleaning pipeline."""

    def test_clean_data_success(self, sample_raw_data):
        """Test successful data cleaning."""
        df_clean = clean_data(sample_raw_data)

        assert len(df_clean) > 0
        assert 'user_id' in df_clean.columns
        assert 'date' in df_clean.columns
        assert df_clean['steps'].notna().all()

    def test_clean_data_removes_invalid_rows(self, sample_raw_data):
        """Test that invalid rows are removed."""
        df = sample_raw_data.copy()
        # Add row with all NaN values except user_id
        df.loc[len(df)] = {
            'user_id': 999,
            'date': pd.NaT,
            'steps': np.nan,
            'calories_burned': np.nan,
            'distance_km': np.nan,
            'active_minutes': np.nan,
            'sleep_hours': np.nan,
            'heart_rate_avg': np.nan,
            'workout_type': None,
            'mood': None,
            'weather_conditions': None,
            'location': None
        }

        df_clean = clean_data(df)

        # Invalid row should be removed
        assert 999 not in df_clean['user_id'].values

    def test_clean_data_date_conversion(self, sample_raw_data):
        """Test that dates are properly converted."""
        df_clean = clean_data(sample_raw_data)

        assert pd.api.types.is_datetime64_any_dtype(df_clean['date'])
