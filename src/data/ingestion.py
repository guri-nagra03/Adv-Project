"""
Data Ingestion Module

Handles CSV file uploads and simulated streaming of wearable device data.
Provides data loading, validation, and streaming simulation for educational purposes.
"""

import time
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

from config.settings import settings
from src.utils.constants import REQUIRED_COLUMNS
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class DataIngestionError(Exception):
    """Raised when data ingestion fails"""
    pass


def load_csv(
    filepath: Optional[str] = None,
    limit: Optional[int] = None,
    skip_validation: bool = False
) -> pd.DataFrame:
    """
    Load wearable data from CSV file.

    Args:
        filepath: Path to CSV file (default: settings.DATASET_PATH)
        limit: Maximum number of rows to load (default: settings.DATA_LIMIT)
        skip_validation: Skip schema validation (default: False)

    Returns:
        DataFrame with wearable data

    Raises:
        DataIngestionError: If file not found or validation fails
    """
    filepath = filepath or settings.DATASET_PATH
    limit = limit or settings.DATA_LIMIT

    logger.info(f"Loading data from {filepath} (limit: {limit} rows)")

    # Check if file exists
    if not Path(filepath).exists():
        raise DataIngestionError(f"Dataset not found at {filepath}")

    try:
        # Load CSV with specified limit
        df = pd.read_csv(filepath, nrows=limit)
        logger.info(f"Loaded {len(df)} rows from {filepath}")

        # Validate schema unless skipped
        if not skip_validation:
            validate_schema(df)
            logger.info("Schema validation passed")

        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Sort by user_id and date
        df = df.sort_values(['user_id', 'date']).reset_index(drop=True)

        logger.info(f"Data ingestion complete: {len(df)} rows, {df['user_id'].nunique()} unique users")
        return df

    except pd.errors.EmptyDataError:
        raise DataIngestionError("CSV file is empty")
    except pd.errors.ParserError as e:
        raise DataIngestionError(f"Failed to parse CSV: {e}")
    except Exception as e:
        raise DataIngestionError(f"Unexpected error loading data: {e}")


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has required columns and correct types.

    Args:
        df: DataFrame to validate

    Returns:
        True if validation passes

    Raises:
        DataIngestionError: If validation fails
    """
    # Check required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise DataIngestionError(
            f"Missing required columns: {missing_columns}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    # Check for empty DataFrame
    if df.empty:
        raise DataIngestionError("DataFrame is empty")

    # Validate numeric columns
    numeric_columns = [
        'steps', 'calories_burned', 'distance_km',
        'active_minutes', 'sleep_hours', 'heart_rate_avg'
    ]

    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Try to convert
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.warning(f"Converted column '{col}' to numeric type")
            except Exception as e:
                raise DataIngestionError(f"Column '{col}' must be numeric: {e}")

    # Validate date column
    try:
        pd.to_datetime(df['date'], errors='coerce')
    except Exception as e:
        raise DataIngestionError(f"Column 'date' must be in valid date format: {e}")

    # Check for all-null columns
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        raise DataIngestionError(f"Columns contain all null values: {null_columns}")

    logger.debug("Schema validation successful")
    return True


def simulate_stream(
    dataframe: pd.DataFrame,
    batch_size: int = 100,
    delay_seconds: float = 5.0
) -> Generator[pd.DataFrame, None, None]:
    """
    Simulate real-time data streaming by yielding batches with delays.

    This is for educational demonstration of how streaming data ingestion
    would work with wearable devices sending data periodically.

    Args:
        dataframe: Source DataFrame to stream
        batch_size: Number of rows per batch
        delay_seconds: Delay between batches (simulates network latency)

    Yields:
        DataFrame batches

    Example:
        >>> df = load_csv()
        >>> for batch in simulate_stream(df, batch_size=50, delay_seconds=2):
        ...     process_batch(batch)
    """
    total_rows = len(dataframe)
    logger.info(
        f"Starting simulated stream: {total_rows} rows, "
        f"batch_size={batch_size}, delay={delay_seconds}s"
    )

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch = dataframe.iloc[start_idx:end_idx].copy()

        logger.debug(f"Yielding batch: rows {start_idx} to {end_idx}")
        yield batch

        # Simulate network delay (skip delay for last batch)
        if end_idx < total_rows:
            time.sleep(delay_seconds)

    logger.info("Simulated stream complete")


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the dataset.

    Args:
        df: DataFrame to summarize

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_rows': len(df),
        'unique_users': df['user_id'].nunique(),
        'date_range': {
            'start': df['date'].min(),
            'end': df['date'].max(),
            'days': (df['date'].max() - df['date'].min()).days
        },
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df[[
            'steps', 'calories_burned', 'distance_km',
            'active_minutes', 'sleep_hours', 'heart_rate_avg'
        ]].describe().to_dict()
    }

    return summary


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter DataFrame by date range.

    Args:
        df: DataFrame to filter
        start_date: Start date (inclusive) in YYYY-MM-DD format
        end_date: End date (inclusive) in YYYY-MM-DD format

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()

    if start_date:
        start = pd.to_datetime(start_date)
        filtered_df = filtered_df[filtered_df['date'] >= start]
        logger.info(f"Filtered to dates >= {start_date}: {len(filtered_df)} rows")

    if end_date:
        end = pd.to_datetime(end_date)
        filtered_df = filtered_df[filtered_df['date'] <= end]
        logger.info(f"Filtered to dates <= {end_date}: {len(filtered_df)} rows")

    return filtered_df


def filter_by_users(
    df: pd.DataFrame,
    user_ids: list[int]
) -> pd.DataFrame:
    """
    Filter DataFrame to specific user IDs.

    Args:
        df: DataFrame to filter
        user_ids: List of user IDs to include

    Returns:
        Filtered DataFrame
    """
    filtered_df = df[df['user_id'].isin(user_ids)].copy()
    logger.info(f"Filtered to {len(user_ids)} users: {len(filtered_df)} rows")
    return filtered_df


# Example usage
if __name__ == "__main__":
    # Load data
    df = load_csv()
    print(f"Loaded {len(df)} rows")

    # Get summary
    summary = get_dataset_summary(df)
    print(f"\nDataset Summary:")
    print(f"  Total rows: {summary['total_rows']}")
    print(f"  Unique users: {summary['unique_users']}")
    print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")

    # Simulate streaming (small batch for demo)
    print("\nSimulating stream (first 200 rows, batch_size=50):")
    for i, batch in enumerate(simulate_stream(df.head(200), batch_size=50, delay_seconds=1)):
        print(f"  Batch {i+1}: {len(batch)} rows")
