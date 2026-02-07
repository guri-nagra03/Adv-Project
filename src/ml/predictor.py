"""
ML Predictor

Loads trained model and performs inference to generate cardiovascular
wellness risk scores for new data.

Outputs:
- Risk probabilities (0-1 scale) for each class
- Combined risk score (probability of medium + high risk)
- Predicted class (0, 1, 2)
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple

from config.settings import settings
from src.ml.trainer import load_model
from src.utils.constants import RISK_LABEL_MAPPING, RISK_LABEL_TO_COLOR
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class PredictionError(Exception):
    """Raised when prediction fails"""
    pass


class RiskPredictor:
    """
    Cardiovascular risk predictor using trained ML model.

    Example:
        >>> predictor = RiskPredictor()
        >>> result = predictor.predict(features_df)
        >>> print(result['risk_score'])  # Combined medium+high probability
    """

    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to model file (default: settings.MODEL_PATH)
            scaler_path: Path to scaler file (default: settings.SCALER_PATH)
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.scaler_path = scaler_path or settings.SCALER_PATH

        logger.info("Loading trained model and scaler...")
        self.model, self.scaler = load_model(self.model_path, self.scaler_path)
        logger.info("Predictor initialized")

    def predict(
        self,
        features: Union[pd.DataFrame, pd.Series, dict]
    ) -> Dict:
        """
        Predict risk for single record or batch.

        Args:
            features: Features as DataFrame, Series, or dict

        Returns:
            Dictionary with prediction results:
            - risk_score: Combined probability (0-1) of medium+high risk
            - risk_probabilities: Dict with probabilities per class
            - predicted_class: Predicted label (0, 1, 2)
            - predicted_label: Human-readable label ("Low", "Medium", "High")
            - risk_color: Color name for risk level

        Example:
            >>> result = predictor.predict({'resting_hr_estimate': 85, ...})
            >>> print(f"Risk score: {result['risk_score']:.2f}")
        """
        # Convert to DataFrame if needed
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()

        # Validate features
        self._validate_features(features_df)

        # Scale features
        features_scaled = self.scaler.transform(features_df)

        # Predict probabilities
        probabilities = self.model.predict_proba(features_scaled)

        # Predict class
        predicted_classes = self.model.predict(features_scaled)

        # Build results
        if len(features_df) == 1:
            # Single prediction
            result = self._build_single_result(
                probabilities[0],
                predicted_classes[0]
            )
        else:
            # Batch prediction
            result = self._build_batch_result(
                probabilities,
                predicted_classes,
                features_df.index
            )

        return result

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk for batch of records (convenience method).

        Args:
            features_df: DataFrame with features

        Returns:
            DataFrame with predictions added as columns
        """
        result_df = features_df.copy()

        # Get predictions
        features_scaled = self.scaler.transform(features_df)
        probabilities = self.model.predict_proba(features_scaled)
        predicted_classes = self.model.predict(features_scaled)

        # Add prediction columns
        result_df['risk_score'] = probabilities[:, 1] + probabilities[:, 2]  # Medium + High
        result_df['predicted_class'] = predicted_classes
        result_df['predicted_label'] = [RISK_LABEL_MAPPING[c] for c in predicted_classes]

        # Add individual class probabilities
        result_df['prob_low'] = probabilities[:, 0]
        result_df['prob_medium'] = probabilities[:, 1]
        result_df['prob_high'] = probabilities[:, 2]

        logger.info(f"Batch prediction complete for {len(result_df)} records")

        return result_df

    def _validate_features(self, features_df: pd.DataFrame):
        """Validate that features DataFrame has required columns."""
        # Get feature names from scaler (order matters)
        if hasattr(self.scaler, 'feature_names_in_'):
            required_features = self.scaler.feature_names_in_.tolist()
        else:
            # Older sklearn version, can't validate
            logger.warning("Cannot validate feature names (sklearn version < 1.0)")
            return

        missing_features = set(required_features) - set(features_df.columns)
        if missing_features:
            raise PredictionError(
                f"Missing required features: {missing_features}. "
                f"Expected: {required_features}"
            )

        # Check for NaN values
        if features_df[required_features].isnull().any().any():
            raise PredictionError(
                "Features contain NaN values. "
                "Please handle missing values before prediction."
            )

    def _build_single_result(
        self,
        probabilities: np.ndarray,
        predicted_class: int
    ) -> Dict:
        """Build result dict for single prediction."""
        # Combined risk score (probability of being medium or high risk)
        risk_score = probabilities[1] + probabilities[2]

        result = {
            'risk_score': float(risk_score),
            'risk_probabilities': {
                'Low': float(probabilities[0]),
                'Medium': float(probabilities[1]),
                'High': float(probabilities[2])
            },
            'predicted_class': int(predicted_class),
            'predicted_label': RISK_LABEL_MAPPING[predicted_class],
            'risk_color': RISK_LABEL_TO_COLOR[RISK_LABEL_MAPPING[predicted_class]]
        }

        logger.debug(
            f"Prediction: {result['predicted_label']} "
            f"(score: {risk_score:.3f})"
        )

        return result

    def _build_batch_result(
        self,
        probabilities: np.ndarray,
        predicted_classes: np.ndarray,
        index: pd.Index
    ) -> pd.DataFrame:
        """Build DataFrame for batch predictions."""
        results = []

        for i, idx in enumerate(index):
            result = {
                'index': idx,
                'risk_score': float(probabilities[i, 1] + probabilities[i, 2]),
                'prob_low': float(probabilities[i, 0]),
                'prob_medium': float(probabilities[i, 1]),
                'prob_high': float(probabilities[i, 2]),
                'predicted_class': int(predicted_classes[i]),
                'predicted_label': RISK_LABEL_MAPPING[predicted_classes[i]],
                'risk_color': RISK_LABEL_TO_COLOR[RISK_LABEL_MAPPING[predicted_classes[i]]]
            }
            results.append(result)

        result_df = pd.DataFrame(results)
        logger.info(f"Batch prediction: {len(result_df)} records processed")

        return result_df


def predict_risk_score(
    features: Union[pd.DataFrame, dict],
    model_path: str = None,
    scaler_path: str = None
) -> Union[float, pd.Series]:
    """
    Convenience function to get risk score directly.

    Args:
        features: Features DataFrame or dict
        model_path: Optional model path
        scaler_path: Optional scaler path

    Returns:
        Risk score (float) or Series of scores

    Example:
        >>> score = predict_risk_score({'resting_hr_estimate': 90, ...})
        >>> print(f"Risk: {score:.2f}")
    """
    predictor = RiskPredictor(model_path, scaler_path)
    result = predictor.predict(features)

    if isinstance(result, dict):
        return result['risk_score']
    else:
        return result['risk_score']


# Example usage
if __name__ == "__main__":
    from src.data.ingestion import load_csv
    from src.data.preprocessing import clean_data
    from src.data.feature_engineering import (
        create_cardiovascular_features,
        get_feature_columns
    )

    # Load and process sample data
    print("Loading sample data...")
    df = load_csv(limit=100)
    df_clean = clean_data(df)
    features_df = create_cardiovascular_features(df_clean)

    # Get feature columns
    feature_cols = get_feature_columns()
    X = features_df[feature_cols].dropna()

    print(f"\nMaking predictions for {len(X)} records...")

    # Initialize predictor
    try:
        predictor = RiskPredictor()

        # Single prediction (first record)
        print("\n=== Single Prediction Example ===")
        single_features = X.iloc[0]
        result = predictor.predict(single_features)

        print(f"Features: {single_features.to_dict()}")
        print(f"\nPrediction:")
        print(f"  Risk Score: {result['risk_score']:.3f}")
        print(f"  Predicted Label: {result['predicted_label']}")
        print(f"  Probabilities:")
        for label, prob in result['risk_probabilities'].items():
            print(f"    {label}: {prob:.3f}")

        # Batch prediction
        print("\n=== Batch Prediction Example ===")
        batch_results = predictor.predict_batch(X.head(10))
        print(batch_results[['predicted_label', 'risk_score', 'prob_low', 'prob_medium', 'prob_high']])

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Model must be trained first. Run trainer.py to train a model.")
