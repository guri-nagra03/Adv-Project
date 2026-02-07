"""
Feature Engineering Module

Derives cardiovascular wellness risk features from raw wearable device data.
These features are used for ML model training and risk assessment.

Core Features:
- resting_hr_estimate: Estimated resting heart rate from low-activity periods
- activity_score: Composite score from steps, active minutes, distance
- sleep_hours_avg: Rolling average of sleep duration
- sedentary_ratio: Proportion of sedentary days
- workout_consistency: Proportion of days with workouts
- hr_variability_proxy: Heart rate variability estimate
- mood_stress_ratio: Proportion of stressed mood entries
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from config.settings import settings
from src.utils.constants import MOOD_STRESSED, FEATURE_NAMES
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class FeatureEngineeringError(Exception):
    """Raised when feature engineering fails"""
    pass


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data to daily level if not already aggregated.

    If data has multiple entries per user per day, aggregate to single entry.

    Args:
        df: Raw DataFrame with potential multiple entries per day

    Returns:
        DataFrame with one row per user per day
    """
    # Check if already at daily granularity
    daily_counts = df.groupby(['user_id', 'date']).size()
    if (daily_counts > 1).any():
        logger.info(
            f"Multiple entries per day detected for some users. "
            f"Aggregating to daily level..."
        )

        # Aggregate numeric columns with mean
        numeric_agg = {
            'steps': 'sum',  # Sum steps across day
            'calories_burned': 'sum',  # Sum calories
            'distance_km': 'sum',  # Sum distance
            'active_minutes': 'sum',  # Sum active minutes
            'sleep_hours': 'mean',  # Average sleep (if multiple recordings)
            'heart_rate_avg': 'mean'  # Average heart rate
        }

        # Aggregate categorical with mode (most common)
        categorical_cols = ['workout_type', 'mood', 'weather_conditions', 'location']
        categorical_agg = {col: lambda x: x.mode()[0] if not x.mode().empty else 'None'
                          for col in categorical_cols if col in df.columns}

        agg_dict = {**numeric_agg, **categorical_agg}

        df_daily = df.groupby(['user_id', 'date']).agg(agg_dict).reset_index()
        logger.info(f"Aggregated to {len(df_daily)} daily records")
        return df_daily

    logger.info("Data already at daily granularity")
    return df.copy()


def create_cardiovascular_features(
    df: pd.DataFrame,
    min_days: int = 7
) -> pd.DataFrame:
    """
    Create comprehensive cardiovascular wellness features.

    Args:
        df: Daily aggregated DataFrame
        min_days: Minimum days of data required per user for rolling features

    Returns:
        DataFrame with additional feature columns
    """
    logger.info(f"Creating cardiovascular features for {len(df)} records")

    # Ensure daily aggregation
    df_daily = aggregate_daily(df)

    # Sort by user and date
    df_daily = df_daily.sort_values(['user_id', 'date']).reset_index(drop=True)

    # Group by user for time-series features
    df_features = df_daily.groupby('user_id', group_keys=False).apply(
        _compute_user_features,
        min_days=min_days
    ).reset_index(drop=True)

    # Remove rows with insufficient data
    initial_rows = len(df_features)
    df_features = df_features.dropna(subset=['activity_score', 'sleep_hours_avg'])
    removed_rows = initial_rows - len(df_features)

    if removed_rows > 0:
        logger.warning(
            f"Removed {removed_rows} rows due to insufficient data for feature calculation"
        )

    logger.info(f"Feature engineering complete: {len(df_features)} records with features")

    return df_features


def _compute_user_features(user_df: pd.DataFrame, min_days: int) -> pd.DataFrame:
    """
    Compute features for a single user's time series.

    Args:
        user_df: DataFrame for single user
        min_days: Minimum days required

    Returns:
        DataFrame with added features
    """
    user_df = user_df.copy()

    # Skip users with insufficient data
    if len(user_df) < min_days:
        logger.debug(
            f"User {user_df['user_id'].iloc[0]}: insufficient data "
            f"({len(user_df)} days < {min_days} required)"
        )
        return user_df

    # 1. Resting Heart Rate Estimate
    user_df['resting_hr_estimate'] = _estimate_resting_hr(
        user_df['heart_rate_avg'],
        user_df['steps']
    )

    # 2. Activity Score (composite)
    user_df['activity_score'] = _calculate_activity_score(
        user_df['steps'],
        user_df['active_minutes'],
        user_df['distance_km']
    )

    # 3. Sleep Hours (7-day rolling average)
    user_df['sleep_hours_avg'] = user_df['sleep_hours'].rolling(
        window=settings.ROLLING_WINDOW_SHORT,
        min_periods=1
    ).mean()

    # 4. Sedentary Ratio (30-day)
    user_df['sedentary_ratio'] = _calculate_sedentary_ratio(
        user_df['steps'],
        window=settings.ROLLING_WINDOW_LONG
    )

    # 5. Workout Consistency (30-day)
    user_df['workout_consistency'] = _calculate_workout_consistency(
        user_df['workout_type'],
        window=settings.ROLLING_WINDOW_LONG
    )

    # 6. Heart Rate Variability Proxy (7-day std dev)
    user_df['hr_variability_proxy'] = user_df['heart_rate_avg'].rolling(
        window=settings.ROLLING_WINDOW_SHORT,
        min_periods=1
    ).std()

    # 7. Mood Stress Ratio (14-day)
    if 'mood' in user_df.columns:
        user_df['mood_stress_ratio'] = _calculate_stress_ratio(
            user_df['mood'],
            window=14
        )
    else:
        user_df['mood_stress_ratio'] = 0.0

    # 8. Additional derived features
    user_df['calories_per_step'] = user_df['calories_burned'] / (user_df['steps'] + 1)
    user_df['avg_hr_to_resting_ratio'] = user_df['heart_rate_avg'] / (user_df['resting_hr_estimate'] + 1)

    return user_df


def _estimate_resting_hr(hr_series: pd.Series, steps_series: pd.Series) -> pd.Series:
    """
    Estimate resting heart rate from low-activity periods.

    Resting HR is approximated as the average HR when steps < threshold.

    Args:
        hr_series: Heart rate time series
        steps_series: Steps time series

    Returns:
        Series with resting HR estimates
    """
    # Create a copy
    resting_hr = hr_series.copy()

    # Filter to low-activity periods
    low_activity_mask = steps_series < settings.RESTING_HR_STEPS_THRESHOLD

    if low_activity_mask.sum() > 0:
        # Use rolling mean of low-activity HR as resting HR estimate
        low_activity_hr = hr_series.where(low_activity_mask)

        # Forward fill resting HR estimate (carry forward last known resting HR)
        resting_hr = low_activity_hr.fillna(method='ffill')

        # Backward fill for initial missing values
        resting_hr = resting_hr.fillna(method='bfill')

        # If still missing, use overall median
        resting_hr = resting_hr.fillna(hr_series.median())
    else:
        # No low-activity periods, use percentile as proxy
        resting_hr = hr_series.quantile(0.25)

    return resting_hr


def _calculate_activity_score(
    steps: pd.Series,
    active_minutes: pd.Series,
    distance_km: pd.Series,
    steps_weight: float = 0.4,
    minutes_weight: float = 0.4,
    distance_weight: float = 0.2
) -> pd.Series:
    """
    Calculate composite activity score from multiple metrics.

    Score components (normalized 0-100):
    - Steps: normalized to 10,000 steps/day
    - Active minutes: normalized to 60 min/day
    - Distance: normalized to 10 km/day

    Args:
        steps: Daily steps
        active_minutes: Daily active minutes
        distance_km: Daily distance in km
        steps_weight: Weight for steps component
        minutes_weight: Weight for active minutes component
        distance_weight: Weight for distance component

    Returns:
        Series with activity scores (0-100+)
    """
    # Normalize each component to 0-100 scale
    steps_score = (steps / 10000) * 100
    minutes_score = (active_minutes / 60) * 100
    distance_score = (distance_km / 10) * 100

    # Weighted combination
    activity_score = (
        steps_score * steps_weight +
        minutes_score * minutes_weight +
        distance_score * distance_weight
    )

    return activity_score


def _calculate_sedentary_ratio(
    steps: pd.Series,
    window: int = 30,
    threshold: int = None
) -> pd.Series:
    """
    Calculate proportion of sedentary days in rolling window.

    A day is considered sedentary if steps < threshold.

    Args:
        steps: Daily steps time series
        window: Rolling window size in days
        threshold: Steps threshold for sedentary day

    Returns:
        Series with sedentary ratios (0-1)
    """
    threshold = threshold or settings.SEDENTARY_STEPS_THRESHOLD

    # Mark sedentary days
    is_sedentary = (steps < threshold).astype(int)

    # Calculate rolling proportion
    sedentary_ratio = is_sedentary.rolling(
        window=window,
        min_periods=1
    ).mean()

    return sedentary_ratio


def _calculate_workout_consistency(
    workout_type: pd.Series,
    window: int = 30
) -> pd.Series:
    """
    Calculate proportion of days with structured workouts.

    Args:
        workout_type: Categorical workout type series
        window: Rolling window size in days

    Returns:
        Series with workout consistency ratios (0-1)
    """
    # Mark days with workout (not 'None' or null)
    has_workout = (~workout_type.isin(['None', '', np.nan])).astype(int)

    # Calculate rolling proportion
    workout_consistency = has_workout.rolling(
        window=window,
        min_periods=1
    ).mean()

    return workout_consistency


def _calculate_stress_ratio(
    mood: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    Calculate proportion of stressed mood entries in rolling window.

    Args:
        mood: Categorical mood series
        window: Rolling window size in days

    Returns:
        Series with stress ratios (0-1)
    """
    # Mark stressed days
    is_stressed = (mood == MOOD_STRESSED).astype(int)

    # Calculate rolling proportion
    stress_ratio = is_stressed.rolling(
        window=window,
        min_periods=1
    ).mean()

    return stress_ratio


def get_feature_columns() -> List[str]:
    """
    Get list of feature column names.

    Returns:
        List of feature names
    """
    return FEATURE_NAMES


def prepare_features_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features DataFrame for ML model input.

    - Select only feature columns
    - Drop rows with any missing features
    - Reset index

    Args:
        df: DataFrame with all columns including features

    Returns:
        Clean features DataFrame ready for ML
    """
    feature_cols = get_feature_columns()

    # Check which features are present
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = set(feature_cols) - set(available_features)

    if missing_features:
        logger.warning(f"Missing features: {missing_features}")

    # Select available features
    features_df = df[available_features].copy()

    # Drop rows with missing values
    initial_rows = len(features_df)
    features_df = features_df.dropna()
    dropped_rows = initial_rows - len(features_df)

    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with missing feature values")

    return features_df


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'user_id': [1] * 30,
        'date': pd.date_range('2023-01-01', periods=30),
        'steps': np.random.randint(3000, 15000, 30),
        'active_minutes': np.random.randint(20, 90, 30),
        'distance_km': np.random.uniform(2, 12, 30),
        'sleep_hours': np.random.uniform(5, 9, 30),
        'heart_rate_avg': np.random.randint(65, 95, 30),
        'calories_burned': np.random.randint(1500, 3000, 30),
        'workout_type': np.random.choice(['Running', 'Cycling', 'None', 'Gym'], 30),
        'mood': np.random.choice(['Happy', 'Neutral', 'Stressed', 'Tired'], 30)
    })

    print("Original data:")
    print(sample_data.head())

    # Create features
    features_df = create_cardiovascular_features(sample_data)

    print("\nFeatures created:")
    print(features_df[['date'] + get_feature_columns()].head(10))

    print("\nFeature summary:")
    print(features_df[get_feature_columns()].describe())
