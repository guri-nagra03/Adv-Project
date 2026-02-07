"""
Data Preprocessing Module

Handles data cleaning, validation, missing value imputation,
and outlier detection for wearable device data.
"""

from typing import Optional, Literal

import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class PreprocessingError(Exception):
    """Raised when preprocessing fails"""
    pass


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Starting data cleaning: {len(df)} rows")

    df_clean = df.copy()

    # Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    if duplicates_removed > 0:
        logger.warning(f"Removed {duplicates_removed} duplicate rows")

    # Validate and clean numeric ranges
    df_clean = validate_ranges(df_clean)

    # Handle missing values
    df_clean = handle_missing_values(df_clean)

    # Detect and handle outliers
    df_clean = detect_outliers(df_clean)

    logger.info(f"Data cleaning complete: {len(df_clean)} rows remaining")
    return df_clean


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that numeric values are within physiologically plausible ranges.
    Set out-of-range values to NaN for later imputation.

    Args:
        df: DataFrame to validate

    Returns:
        DataFrame with invalid values set to NaN
    """
    df_validated = df.copy()

    # Define valid ranges
    ranges = {
        'sleep_hours': (0, 24),
        'heart_rate_avg': (30, 250),  # Extreme but physiologically possible
        'steps': (0, 50000),  # Upper limit for extremely active individuals
        'active_minutes': (0, 1440),  # Max minutes in a day
        'distance_km': (0, 100),  # Upper limit for ultra-endurance
        'calories_burned': (0, 10000)  # Upper limit for extreme activity
    }

    for col, (min_val, max_val) in ranges.items():
        if col in df_validated.columns:
            # Count out-of-range values
            out_of_range = (
                (df_validated[col] < min_val) |
                (df_validated[col] > max_val)
            ).sum()

            if out_of_range > 0:
                logger.warning(
                    f"{col}: {out_of_range} values out of range "
                    f"[{min_val}, {max_val}] - setting to NaN"
                )
                # Set out-of-range to NaN
                df_validated.loc[
                    (df_validated[col] < min_val) | (df_validated[col] > max_val),
                    col
                ] = np.nan

    return df_validated


def handle_missing_values(
    df: pd.DataFrame,
    strategy: Literal['drop', 'zero', 'median', 'forward_fill', 'smart'] = 'smart'
) -> pd.DataFrame:
    """
    Handle missing values using specified strategy.

    Args:
        df: DataFrame with missing values
        strategy:
            - 'drop': Drop rows with any missing values
            - 'zero': Fill with 0 (for activity metrics)
            - 'median': Fill with median (for all numeric)
            - 'forward_fill': Forward fill (for time series)
            - 'smart': Different strategy per column (recommended)

    Returns:
        DataFrame with missing values handled
    """
    df_filled = df.copy()
    missing_before = df_filled.isnull().sum().sum()

    if missing_before == 0:
        logger.info("No missing values found")
        return df_filled

    logger.info(f"Handling {missing_before} missing values (strategy: {strategy})")

    if strategy == 'drop':
        df_filled = df_filled.dropna()

    elif strategy == 'zero':
        df_filled = df_filled.fillna(0)

    elif strategy == 'median':
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_filled[col].isnull().any():
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())

    elif strategy == 'forward_fill':
        df_filled = df_filled.fillna(method='ffill')

    elif strategy == 'smart':
        # Activity metrics: assume 0 if missing (no activity recorded)
        activity_cols = ['steps', 'active_minutes', 'distance_km']
        for col in activity_cols:
            if col in df_filled.columns and df_filled[col].isnull().any():
                fill_count = df_filled[col].isnull().sum()
                df_filled[col] = df_filled[col].fillna(0)
                logger.debug(f"{col}: filled {fill_count} missing values with 0")

        # Sleep hours: use median for user, or global median
        if 'sleep_hours' in df_filled.columns and df_filled['sleep_hours'].isnull().any():
            df_filled['sleep_hours'] = df_filled.groupby('user_id')['sleep_hours'].transform(
                lambda x: x.fillna(x.median() if x.notna().any() else df_filled['sleep_hours'].median())
            )
            logger.debug("sleep_hours: filled with user median")

        # Heart rate: forward fill (most recent valid measurement)
        if 'heart_rate_avg' in df_filled.columns and df_filled['heart_rate_avg'].isnull().any():
            df_filled['heart_rate_avg'] = df_filled.groupby('user_id')['heart_rate_avg'].transform(
                lambda x: x.fillna(method='ffill')
            )
            # If still missing (first entries), use median
            df_filled['heart_rate_avg'] = df_filled['heart_rate_avg'].fillna(
                df_filled['heart_rate_avg'].median()
            )
            logger.debug("heart_rate_avg: filled with forward fill + median")

        # Calories: use median
        if 'calories_burned' in df_filled.columns and df_filled['calories_burned'].isnull().any():
            df_filled['calories_burned'] = df_filled['calories_burned'].fillna(
                df_filled['calories_burned'].median()
            )
            logger.debug("calories_burned: filled with median")

        # Workout type: fill with 'None'
        if 'workout_type' in df_filled.columns and df_filled['workout_type'].isnull().any():
            df_filled['workout_type'] = df_filled['workout_type'].fillna('None')
            logger.debug("workout_type: filled with 'None'")

        # Mood: fill with 'Neutral'
        if 'mood' in df_filled.columns and df_filled['mood'].isnull().any():
            df_filled['mood'] = df_filled['mood'].fillna('Neutral')
            logger.debug("mood: filled with 'Neutral'")

    missing_after = df_filled.isnull().sum().sum()
    logger.info(f"Missing values handled: {missing_before} → {missing_after}")

    return df_filled


def detect_outliers(
    df: pd.DataFrame,
    method: Literal['iqr', 'zscore', 'none'] = 'iqr',
    handle: Literal['cap', 'remove', 'flag'] = 'cap'
) -> pd.DataFrame:
    """
    Detect and handle outliers in numeric columns.

    Args:
        df: DataFrame to process
        method:
            - 'iqr': Interquartile range method
            - 'zscore': Z-score method (3 standard deviations)
            - 'none': No outlier detection
        handle:
            - 'cap': Cap outliers at threshold
            - 'remove': Remove rows with outliers
            - 'flag': Add flag column but keep data

    Returns:
        DataFrame with outliers handled
    """
    if method == 'none':
        return df

    df_processed = df.copy()
    numeric_cols = ['steps', 'calories_burned', 'distance_km',
                    'active_minutes', 'sleep_hours', 'heart_rate_avg']

    outlier_counts = {}

    for col in numeric_cols:
        if col not in df_processed.columns:
            continue

        if method == 'iqr':
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (
                (df_processed[col] < lower_bound) |
                (df_processed[col] > upper_bound)
            )

        elif method == 'zscore':
            z_scores = np.abs(
                (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
            )
            outliers = z_scores > 3

        outlier_count = outliers.sum()
        outlier_counts[col] = outlier_count

        if outlier_count > 0:
            if handle == 'cap':
                # Cap at bounds
                if method == 'iqr':
                    df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
                    df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
                logger.debug(f"{col}: capped {outlier_count} outliers")

            elif handle == 'remove':
                df_processed = df_processed[~outliers]
                logger.debug(f"{col}: removed {outlier_count} outlier rows")

            elif handle == 'flag':
                flag_col = f"{col}_outlier"
                df_processed[flag_col] = outliers
                logger.debug(f"{col}: flagged {outlier_count} outliers in {flag_col}")

    total_outliers = sum(outlier_counts.values())
    if total_outliers > 0:
        logger.info(
            f"Outlier detection ({method}, {handle}): "
            f"found {total_outliers} outliers across {len(outlier_counts)} columns"
        )

    return df_processed


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features from date column.

    Args:
        df: DataFrame with 'date' column

    Returns:
        DataFrame with additional time features
    """
    df_time = df.copy()

    if 'date' not in df_time.columns:
        logger.warning("'date' column not found, skipping time feature extraction")
        return df_time

    # Ensure date is datetime
    df_time['date'] = pd.to_datetime(df_time['date'])

    # Extract time features
    df_time['day_of_week'] = df_time['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df_time['is_weekend'] = df_time['day_of_week'].isin([5, 6]).astype(int)
    df_time['month'] = df_time['date'].dt.month
    df_time['week_of_year'] = df_time['date'].dt.isocalendar().week

    logger.debug("Added time features: day_of_week, is_weekend, month, week_of_year")
    return df_time


# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 2],
        'date': pd.date_range('2023-01-01', periods=6),
        'steps': [10000, np.nan, 8000, 50000, 12000, np.nan],  # Outlier at 50000
        'sleep_hours': [7, 8, np.nan, 6, 5, 7],
        'heart_rate_avg': [75, 80, 300, 70, 85, 78],  # Outlier at 300
        'active_minutes': [60, 45, 30, 1500, 50, 40],  # Outlier at 1500
        'calories_burned': [2000, 2100, np.nan, 2200, 1900, 2050],
        'distance_km': [8, 7, 6, 9, 8, 7],
        'workout_type': ['Running', 'Cycling', None, 'Running', 'Gym', 'Walking'],
        'mood': ['Happy', 'Happy', 'Neutral', 'Stressed', None, 'Tired']
    })

    print("Original data:")
    print(sample_data)
    print(f"\nMissing values:\n{sample_data.isnull().sum()}")

    # Clean data
    clean_df = clean_data(sample_data)
    print("\nCleaned data:")
    print(clean_df)
    print(f"\nMissing values after cleaning:\n{clean_df.isnull().sum()}")
