"""
Unit tests for FHIR converter module.
"""

import pytest
import pandas as pd

from src.fhir.converter import (
    create_observation,
    convert_row_to_observations,
    batch_convert_observations
)
from src.utils.constants import LOINC_CODES


class TestCreateObservation:
    """Tests for create_observation function."""

    def test_create_observation_basic(self):
        """Test basic observation creation."""
        obs = create_observation(
            user_id=123,
            date='2023-01-01',
            metric_name='steps',
            value=10000
        )

        assert obs.resourceType == 'Observation'
        assert obs.status == 'final'
        assert obs.subject.reference == 'Patient/123'
        assert obs.valueQuantity.value == 10000

    def test_loinc_code_mapping(self):
        """Test LOINC code mapping."""
        obs = create_observation(
            user_id=123,
            date='2023-01-01',
            metric_name='heart_rate_avg',
            value=75
        )

        assert obs.code.coding[0].system == 'http://loinc.org'
        assert obs.code.coding[0].code == LOINC_CODES['heart_rate_avg']

    def test_datetime_conversion(self):
        """Test datetime conversion."""
        obs = create_observation(
            user_id=123,
            date='2023-01-15',
            metric_name='steps',
            value=8000
        )

        assert '2023-01-15' in obs.effectiveDateTime

    def test_unknown_metric_raises_error(self):
        """Test that unknown metric raises error."""
        with pytest.raises(Exception):
            create_observation(
                user_id=123,
                date='2023-01-01',
                metric_name='unknown_metric',
                value=100
            )

    def test_all_known_metrics(self):
        """Test all known metrics can be converted."""
        for metric_name in LOINC_CODES.keys():
            obs = create_observation(
                user_id=123,
                date='2023-01-01',
                metric_name=metric_name,
                value=100
            )

            assert obs is not None
            assert obs.code.coding[0].code == LOINC_CODES[metric_name]


class TestConvertRowToObservations:
    """Tests for convert_row_to_observations function."""

    def test_convert_row_basic(self, sample_raw_data):
        """Test converting single row to observations."""
        row = sample_raw_data.iloc[0]

        observations = convert_row_to_observations(row)

        assert len(observations) > 0
        assert all(obs.resourceType == 'Observation' for obs in observations)

    def test_convert_row_specific_metrics(self, sample_raw_data):
        """Test converting specific metrics only."""
        row = sample_raw_data.iloc[0]

        observations = convert_row_to_observations(
            row,
            metrics=['steps', 'heart_rate_avg']
        )

        assert len(observations) == 2

    def test_convert_row_handles_missing(self, sample_raw_data):
        """Test handling of missing values."""
        row = sample_raw_data.iloc[0].copy()
        row['steps'] = None

        observations = convert_row_to_observations(
            row,
            metrics=['steps', 'heart_rate_avg']
        )

        # Should skip steps, only return heart_rate_avg
        assert len(observations) == 1


class TestBatchConvertObservations:
    """Tests for batch_convert_observations function."""

    def test_batch_convert(self, sample_raw_data):
        """Test batch conversion."""
        observations = batch_convert_observations(sample_raw_data)

        assert len(observations) > 0
        # Should have multiple observations per row
        assert len(observations) > len(sample_raw_data)

    def test_batch_convert_specific_metrics(self, sample_raw_data):
        """Test batch conversion with specific metrics."""
        observations = batch_convert_observations(
            sample_raw_data,
            metrics=['steps']
        )

        # Should have one observation per row
        assert len(observations) == len(sample_raw_data)
