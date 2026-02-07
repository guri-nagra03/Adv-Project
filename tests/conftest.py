"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_raw_data():
    """Generate sample raw fitness tracker data."""
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')

    data = {
        'user_id': [1] * 30,
        'date': dates,
        'steps': np.random.randint(3000, 15000, 30),
        'calories_burned': np.random.randint(1500, 3000, 30),
        'distance_km': np.random.uniform(2, 12, 30),
        'active_minutes': np.random.randint(20, 120, 30),
        'sleep_hours': np.random.uniform(5, 9, 30),
        'heart_rate_avg': np.random.randint(60, 95, 30),
        'workout_type': np.random.choice(['Running', 'Cycling', 'Walking', None], 30),
        'mood': np.random.choice(['Happy', 'Stressed', 'Neutral'], 30),
        'weather_conditions': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], 30),
        'location': ['City'] * 30
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_features():
    """Generate sample engineered features."""
    return pd.DataFrame({
        'user_id': [1, 1, 2],
        'date': pd.to_datetime(['2023-01-15', '2023-01-16', '2023-01-15']),
        'resting_hr_estimate': [72.0, 75.0, 85.0],
        'activity_score': [65.0, 70.0, 45.0],
        'sleep_hours_avg': [7.5, 7.2, 6.0],
        'sedentary_ratio': [0.2, 0.25, 0.6],
        'workout_consistency': [0.7, 0.75, 0.4],
        'hr_variability_proxy': [8.0, 9.0, 12.0],
        'mood_stress_ratio': [0.1, 0.15, 0.4]
    })


@pytest.fixture
def sample_labels():
    """Generate sample labels."""
    return pd.Series([0, 0, 2], name='risk_label')


@pytest.fixture
def temp_db_path(tmp_path):
    """Temporary database path for testing."""
    return str(tmp_path / "test_cardioguard.db")


@pytest.fixture
def mock_fhir_client():
    """Mock FHIR client for testing."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.post_observation.return_value = "obs-123"
    client.post_risk_assessment.return_value = "risk-123"
    client.post_flag.return_value = "flag-123"
    client.check_server_status.return_value = True

    return client
