"""
ML Model Trainer

Trains interpretable machine learning models for cardiovascular wellness
risk prediction using synthetic labels.

Supports:
- Logistic Regression (multinomial) - Default, interpretable
- Decision Tree Classifier - Alternative for tree visualization

Models are trained on synthetic labels generated from rule-based logic,
not real patient outcomes. This is for educational demonstration only.
"""

import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from config.settings import settings
from src.utils.constants import RISK_LABEL_MAPPING
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class ModelTrainingError(Exception):
    """Raised when model training fails"""
    pass


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'logistic',
    test_size: float = None,
    random_state: int = None
) -> Tuple[object, StandardScaler, Dict]:
    """
    Train cardiovascular risk prediction model.

    Args:
        X: Features DataFrame
        y: Labels Series (0, 1, 2)
        model_type: 'logistic' or 'decision_tree'
        test_size: Proportion for test split (default: from settings)
        random_state: Random seed (default: from settings)

    Returns:
        Tuple of (trained_model, scaler, evaluation_metrics)

    Example:
        >>> model, scaler, metrics = train_model(features, labels)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    test_size = test_size or settings.ML_TEST_SIZE
    random_state = random_state or settings.RANDOM_SEED

    logger.info(f"Training {model_type} model on {len(X)} samples")

    # Validate inputs
    if len(X) != len(y):
        raise ModelTrainingError(f"Feature and label lengths don't match: {len(X)} vs {len(y)}")

    if len(X) < 100:
        raise ModelTrainingError(f"Insufficient data for training: {len(X)} samples (need >= 100)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain label distribution
    )

    logger.info(
        f"Data split: {len(X_train)} train, {len(X_test)} test "
        f"(test_size={test_size})"
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Features scaled using StandardScaler")

    # Create and train model
    if model_type == 'logistic':
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=settings.ML_MAX_ITER,
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance
        )
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            max_depth=5,  # Limit depth for interpretability
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
            class_weight='balanced'
        )
    else:
        raise ModelTrainingError(f"Unknown model type: {model_type}")

    logger.info(f"Training {model_type} model...")
    model.fit(X_train_scaled, y_train)
    logger.info("Model training complete")

    # Evaluate model
    metrics = evaluate_model(model, X_test_scaled, y_test, X_train_scaled, y_train)

    # Cross-validation score
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=5, scoring='accuracy'
    )
    metrics['cv_accuracy_mean'] = cv_scores.mean()
    metrics['cv_accuracy_std'] = cv_scores.std()

    logger.info(
        f"Cross-validation accuracy: {cv_scores.mean():.3f} "
        f"(±{cv_scores.std():.3f})"
    )

    return model, scaler, metrics


def evaluate_model(
    model: object,
    X_test: np.ndarray,
    y_test: pd.Series,
    X_train: np.ndarray = None,
    y_train: pd.Series = None
) -> Dict:
    """
    Evaluate trained model on test (and optionally train) data.

    Args:
        model: Trained model
        X_test: Test features (scaled)
        y_test: Test labels
        X_train: Training features (scaled), optional
        y_train: Training labels, optional

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model performance...")

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_test, y_pred, average=None)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Build metrics dict
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'per_class_metrics': {
            RISK_LABEL_MAPPING[i]: {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1_score': f1_per_class[i],
                'support': support_per_class[i]
            }
            for i in range(len(support_per_class))
        }
    }

    # Training accuracy (if provided)
    if X_train is not None and y_train is not None:
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        metrics['train_accuracy'] = train_accuracy
        logger.info(f"Training accuracy: {train_accuracy:.3f}")

    # Log results
    logger.info(f"Test accuracy: {accuracy:.3f}")
    logger.info(f"Weighted precision: {precision:.3f}")
    logger.info(f"Weighted recall: {recall:.3f}")
    logger.info(f"Weighted F1: {f1:.3f}")

    logger.info("\nPer-class performance:")
    for label_value, class_metrics in metrics['per_class_metrics'].items():
        logger.info(
            f"  {label_value}: "
            f"P={class_metrics['precision']:.3f}, "
            f"R={class_metrics['recall']:.3f}, "
            f"F1={class_metrics['f1_score']:.3f}, "
            f"Support={class_metrics['support']}"
        )

    logger.info(f"\nConfusion Matrix:\n{cm}")

    return metrics


def save_model(
    model: object,
    scaler: StandardScaler,
    filepath: str = None,
    scaler_path: str = None
) -> Tuple[str, str]:
    """
    Save trained model and scaler to disk.

    Args:
        model: Trained model
        scaler: Fitted scaler
        filepath: Path to save model (default: settings.MODEL_PATH)
        scaler_path: Path to save scaler (default: settings.SCALER_PATH)

    Returns:
        Tuple of (model_path, scaler_path)
    """
    model_path = filepath or settings.MODEL_PATH
    scaler_path = scaler_path or settings.SCALER_PATH

    # Create directories if needed
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")

    # Save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")

    return model_path, scaler_path


def load_model(
    filepath: str = None,
    scaler_path: str = None
) -> Tuple[object, StandardScaler]:
    """
    Load trained model and scaler from disk.

    Args:
        filepath: Path to model file (default: settings.MODEL_PATH)
        scaler_path: Path to scaler file (default: settings.SCALER_PATH)

    Returns:
        Tuple of (model, scaler)

    Raises:
        ModelTrainingError: If files not found
    """
    model_path = filepath or settings.MODEL_PATH
    scaler_path = scaler_path or settings.SCALER_PATH

    if not Path(model_path).exists():
        raise ModelTrainingError(f"Model file not found: {model_path}")

    if not Path(scaler_path).exists():
        raise ModelTrainingError(f"Scaler file not found: {scaler_path}")

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"Scaler loaded from {scaler_path}")

    return model, scaler


def get_feature_importance(
    model: object,
    feature_names: list
) -> pd.DataFrame:
    """
    Extract feature importance from trained model.

    Args:
        model: Trained model (LogisticRegression or DecisionTreeClassifier)
        feature_names: List of feature names

    Returns:
        DataFrame with feature importance scores
    """
    if isinstance(model, LogisticRegression):
        # For multinomial logistic regression, use average absolute coefficients
        coefficients = np.abs(model.coef_).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': coefficients
        }).sort_values('importance', ascending=False)

    elif isinstance(model, DecisionTreeClassifier):
        # For decision tree, use built-in feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

    else:
        logger.warning(f"Feature importance not supported for {type(model)}")
        return pd.DataFrame()

    logger.info("Top 5 most important features:")
    for _, row in importance_df.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return importance_df


# Example usage
if __name__ == "__main__":
    from src.data.ingestion import load_csv
    from src.data.preprocessing import clean_data
    from src.data.feature_engineering import create_cardiovascular_features
    from src.ml.label_generator import generate_synthetic_labels

    # Load and process data
    print("Loading data...")
    df = load_csv(limit=5000)

    print("Cleaning data...")
    df_clean = clean_data(df)

    print("Creating features...")
    features_df = create_cardiovascular_features(df_clean)

    print("Generating labels...")
    labels = generate_synthetic_labels(features_df)

    # Prepare features for ML
    from src.data.feature_engineering import get_feature_columns
    feature_cols = get_feature_columns()
    X = features_df[feature_cols].copy()
    y = labels

    # Remove rows with missing features
    valid_mask = ~X.isnull().any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"\nTraining on {len(X)} samples with {len(feature_cols)} features")
    print(f"Label distribution:\n{y.value_counts().sort_index()}\n")

    # Train model
    model, scaler, metrics = train_model(X, y, model_type='logistic')

    print(f"\nModel Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")

    # Feature importance
    print("\nFeature Importance:")
    importance = get_feature_importance(model, feature_cols)
    print(importance)

    # Save model
    print("\nSaving model...")
    save_model(model, scaler)
    print("Model training complete!")
