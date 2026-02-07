"""
Synthetic Label Generator

Creates educational risk labels for ML model training based on
cardiovascular wellness patterns and domain knowledge.

Since we lack ground truth cardiovascular outcomes, we generate synthetic
labels using rule-based logic to identify patterns associated with elevated
wellness risk.

Label Classes:
- 0: Low Risk - Wellness indicators within expected ranges
- 1: Medium Risk - Some indicators show room for improvement
- 2: High Risk - Multiple indicators warrant attention
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple

from config.settings import settings
from src.utils.constants import RISK_LABEL_MAPPING
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class LabelGenerationError(Exception):
    """Raised when label generation fails"""
    pass


def generate_synthetic_labels(
    features_df: pd.DataFrame,
    config_path: str = None
) -> pd.Series:
    """
    Generate synthetic risk labels based on cardiovascular features.

    Uses rule-based logic from risk_thresholds.yaml to classify each
    record as Low (0), Medium (1), or High (2) risk.

    Args:
        features_df: DataFrame with cardiovascular features
        config_path: Path to risk thresholds config (default: from settings)

    Returns:
        Series with integer labels (0, 1, 2)

    Example:
        >>> features = create_cardiovascular_features(df)
        >>> labels = generate_synthetic_labels(features)
        >>> print(labels.value_counts())
        0    5000  # Low risk
        1    3000  # Medium risk
        2    2000  # High risk
    """
    config_path = config_path or settings.RISK_THRESHOLDS_CONFIG
    logger.info(f"Generating synthetic labels for {len(features_df)} records")

    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        label_rules = config['synthetic_labels']
    except Exception as e:
        raise LabelGenerationError(f"Failed to load label configuration: {e}")

    # Initialize all as low risk (0)
    labels = pd.Series(0, index=features_df.index, name='risk_label')

    # Apply medium risk conditions (label = 1)
    medium_mask = _evaluate_conditions(
        features_df,
        label_rules['medium_risk']['conditions']
    )
    labels[medium_mask] = 1

    # Apply high risk conditions (label = 2)
    # High risk overrides medium
    high_mask = _evaluate_conditions(
        features_df,
        label_rules['high_risk']['conditions']
    )
    labels[high_mask] = 2

    # Log distribution
    label_counts = labels.value_counts().sort_index()
    logger.info("Label distribution:")
    for label_value, count in label_counts.items():
        label_name = RISK_LABEL_MAPPING[label_value]
        percentage = (count / len(labels)) * 100
        logger.info(f"  {label_name} ({label_value}): {count} ({percentage:.1f}%)")

    # Validate distribution (ensure reasonable balance)
    validate_label_distribution(labels)

    return labels


def _evaluate_conditions(
    features_df: pd.DataFrame,
    conditions: list
) -> pd.Series:
    """
    Evaluate a list of conditions and return mask for rows matching ANY condition.

    Args:
        features_df: DataFrame with features
        conditions: List of condition dictionaries

    Returns:
        Boolean Series indicating which rows match any condition
    """
    # Initialize mask (all False)
    combined_mask = pd.Series(False, index=features_df.index)

    for condition in conditions:
        criterion = condition['criterion']
        operator = condition['operator']
        value = condition['value']

        # Check if feature exists
        if criterion not in features_df.columns:
            logger.warning(f"Feature '{criterion}' not found in DataFrame, skipping condition")
            continue

        # Evaluate condition based on operator
        if operator == '>':
            mask = features_df[criterion] > value
        elif operator == '<':
            mask = features_df[criterion] < value
        elif operator == '>=':
            mask = features_df[criterion] >= value
        elif operator == '<=':
            mask = features_df[criterion] <= value
        elif operator == '==':
            mask = features_df[criterion] == value
        elif operator == 'between':
            if isinstance(value, list) and len(value) == 2:
                mask = (features_df[criterion] >= value[0]) & (features_df[criterion] <= value[1])
            else:
                logger.warning(f"Invalid 'between' value for {criterion}: {value}")
                continue
        else:
            logger.warning(f"Unknown operator '{operator}' for {criterion}")
            continue

        # Combine with OR logic (ANY condition triggers)
        combined_mask = combined_mask | mask

        # Log how many records matched this condition
        matched_count = mask.sum()
        if matched_count > 0:
            logger.debug(
                f"Condition matched: {criterion} {operator} {value} "
                f"→ {matched_count} records"
            )

    return combined_mask


def validate_label_distribution(labels: pd.Series) -> bool:
    """
    Validate that label distribution is reasonable for ML training.

    Checks:
    - All labels present (0, 1, 2)
    - No class has < 5% of samples (severe imbalance)
    - No class has > 80% of samples (insufficient variation)

    Args:
        labels: Series with integer labels

    Returns:
        True if valid

    Raises:
        LabelGenerationError: If distribution is problematic
    """
    total = len(labels)
    label_counts = labels.value_counts()

    # Check all labels present
    expected_labels = {0, 1, 2}
    present_labels = set(label_counts.index)
    missing_labels = expected_labels - present_labels

    if missing_labels:
        raise LabelGenerationError(
            f"Missing label classes: {missing_labels}. "
            f"All three risk levels (0, 1, 2) must be present for training."
        )

    # Check for severe imbalance
    for label_value in expected_labels:
        count = label_counts.get(label_value, 0)
        percentage = (count / total) * 100

        if percentage < 5:
            logger.warning(
                f"Label {label_value} ({RISK_LABEL_MAPPING[label_value]}) "
                f"has only {percentage:.1f}% of samples. "
                f"Consider adjusting label rules for better balance."
            )

        if percentage > 80:
            logger.warning(
                f"Label {label_value} ({RISK_LABEL_MAPPING[label_value]}) "
                f"has {percentage:.1f}% of samples. "
                f"Extremely imbalanced - model may not learn effectively."
            )

    logger.info("Label distribution validation passed")
    return True


def compute_label_statistics(
    features_df: pd.DataFrame,
    labels: pd.Series
) -> Dict[str, pd.DataFrame]:
    """
    Compute statistics for each label class.

    Useful for understanding what feature patterns define each risk level.

    Args:
        features_df: DataFrame with features
        labels: Series with labels

    Returns:
        Dictionary mapping label names to feature statistics DataFrames
    """
    stats = {}

    for label_value in [0, 1, 2]:
        label_name = RISK_LABEL_MAPPING[label_value]
        mask = labels == label_value

        if mask.sum() == 0:
            logger.warning(f"No samples for {label_name}")
            continue

        # Get feature statistics for this label
        label_stats = features_df[mask].describe().T
        label_stats['count'] = mask.sum()

        stats[label_name] = label_stats

        logger.debug(f"{label_name} feature statistics:")
        logger.debug(f"\n{label_stats[['mean', 'std', 'min', 'max']]}")

    return stats


def get_label_explanations(
    features_df: pd.DataFrame,
    labels: pd.Series,
    sample_size: int = 5
) -> pd.DataFrame:
    """
    Get sample explanations for why records received specific labels.

    Args:
        features_df: DataFrame with features
        labels: Series with labels
        sample_size: Number of samples per label class

    Returns:
        DataFrame with sample records and their features
    """
    samples = []

    for label_value in [0, 1, 2]:
        label_name = RISK_LABEL_MAPPING[label_value]
        mask = labels == label_value

        if mask.sum() == 0:
            continue

        # Get random samples
        sample_indices = features_df[mask].sample(
            n=min(sample_size, mask.sum()),
            random_state=settings.RANDOM_SEED
        ).index

        for idx in sample_indices:
            sample = {
                'label': label_name,
                'label_value': label_value,
                **features_df.loc[idx].to_dict()
            }
            samples.append(sample)

    return pd.DataFrame(samples)


def analyze_feature_importance_for_labels(
    features_df: pd.DataFrame,
    labels: pd.Series
) -> pd.DataFrame:
    """
    Analyze which features most strongly correlate with label assignment.

    Args:
        features_df: DataFrame with features
        labels: Series with labels

    Returns:
        DataFrame with feature correlations to labels
    """
    # Calculate correlation between each feature and labels
    correlations = []

    for col in features_df.columns:
        if pd.api.types.is_numeric_dtype(features_df[col]):
            corr = features_df[col].corr(labels)
            correlations.append({
                'feature': col,
                'correlation_with_risk': corr,
                'abs_correlation': abs(corr)
            })

    corr_df = pd.DataFrame(correlations).sort_values(
        'abs_correlation',
        ascending=False
    )

    logger.info("Top features correlated with risk labels:")
    for _, row in corr_df.head(5).iterrows():
        logger.info(
            f"  {row['feature']}: {row['correlation_with_risk']:.3f}"
        )

    return corr_df


# Example usage
if __name__ == "__main__":
    from src.data.ingestion import load_csv
    from src.data.preprocessing import clean_data
    from src.data.feature_engineering import create_cardiovascular_features

    # Load and process data
    print("Loading data...")
    df = load_csv(limit=1000)

    print("Cleaning data...")
    df_clean = clean_data(df)

    print("Creating features...")
    features_df = create_cardiovascular_features(df_clean)

    print("\nGenerating synthetic labels...")
    labels = generate_synthetic_labels(features_df)

    print("\nLabel distribution:")
    print(labels.value_counts().sort_index())

    print("\nLabel statistics by class:")
    stats = compute_label_statistics(features_df, labels)
    for label_name, label_stats in stats.items():
        print(f"\n{label_name}:")
        print(label_stats[['mean', 'std']].round(2))

    print("\nFeature correlations with risk:")
    corr_df = analyze_feature_importance_for_labels(features_df, labels)
    print(corr_df.head(10))
