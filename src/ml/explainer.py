"""
SHAP Explainer

Provides interpretable explanations for ML model predictions using
SHAP (SHapley Additive exPlanations) values.

Key Features:
- Identify top contributing features for each prediction
- Generate human-readable explanations
- Visualize feature importance
- Support for logistic regression and decision tree models
"""

import numpy as np
import pandas as pd
import shap
import yaml
from typing import Dict, List, Tuple, Optional

from config.settings import settings
from src.ml.trainer import load_model
from src.utils.constants import HEALTHY_RANGES
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class ExplainerError(Exception):
    """Raised when explanation generation fails"""
    pass


class RiskExplainer:
    """
    Generate interpretable explanations for cardiovascular risk predictions.

    Example:
        >>> explainer = RiskExplainer()
        >>> explanation = explainer.explain(features, patient_id=123)
        >>> print(explanation['summary'])
    """

    def __init__(
        self,
        model_path: str = None,
        scaler_path: str = None,
        config_path: str = None
    ):
        """
        Initialize explainer with trained model.

        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
            config_path: Path to risk thresholds config
        """
        logger.info("Initializing SHAP explainer...")

        # Load model and scaler
        self.model, self.scaler = load_model(model_path, scaler_path)

        # Load configuration for explanations
        config_path = config_path or settings.RISK_THRESHOLDS_CONFIG
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.feature_explanations = config.get('feature_explanations', {})
        self.shap_templates = config.get('shap_templates', {})

        # Initialize SHAP explainer
        # Use background data from training set (or a sample)
        # For now, we'll create explainer without background data
        # In production, you'd want to provide representative background data
        try:
            self.shap_explainer = shap.Explainer(self.model)
            logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None

    def explain(
        self,
        features: pd.DataFrame,
        patient_id: Optional[int] = None,
        top_n: int = None
    ) -> Dict:
        """
        Generate explanation for a single prediction.

        Args:
            features: Features DataFrame (single row or Series)
            patient_id: Optional patient identifier
            top_n: Number of top features to include (default: from settings)

        Returns:
            Dictionary with explanation:
            - patient_id: Patient identifier
            - top_features: List of top contributing features
            - shap_values: Dict of SHAP values per feature
            - summary: Human-readable summary
            - recommendations: List of recommendations
        """
        top_n = top_n or settings.SHAP_TOP_FEATURES

        # Ensure single row
        if isinstance(features, pd.Series):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()

        if len(features_df) > 1:
            logger.warning("Multiple rows provided, using first row only")
            features_df = features_df.iloc[[0]]

        # Scale features
        features_scaled = self.scaler.transform(features_df)

        # Get SHAP values
        if self.shap_explainer is not None:
            try:
                shap_values = self._compute_shap_values(features_scaled)
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}. Using fallback method.")
                shap_values = self._fallback_feature_importance(features_df)
        else:
            shap_values = self._fallback_feature_importance(features_df)

        # Get feature names
        if hasattr(self.scaler, 'feature_names_in_'):
            feature_names = self.scaler.feature_names_in_.tolist()
        else:
            feature_names = features_df.columns.tolist()

        # Create feature importance dict
        feature_importance = {
            feature_names[i]: float(shap_values[i])
            for i in range(len(feature_names))
        }

        # Get top N features by absolute SHAP value
        top_features = self._get_top_features(
            feature_importance,
            features_df.iloc[0],
            top_n
        )

        # Generate human-readable summary
        summary = self._generate_summary(top_features, features_df.iloc[0])

        explanation = {
            'patient_id': patient_id,
            'top_features': top_features,
            'shap_values': feature_importance,
            'summary': summary,
            'feature_values': features_df.iloc[0].to_dict()
        }

        logger.debug(f"Explanation generated for patient {patient_id}")

        return explanation

    def _compute_shap_values(self, features_scaled: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values using SHAP library.

        Args:
            features_scaled: Scaled features array

        Returns:
            Array of SHAP values
        """
        # Get SHAP values
        shap_values_all = self.shap_explainer(features_scaled)

        # For multiclass, we want the SHAP values for the predicted class
        # Or average across classes
        if len(shap_values_all.shape) > 2:
            # Average SHAP values across classes
            shap_values = np.mean(np.abs(shap_values_all.values), axis=1)[0]
        else:
            shap_values = shap_values_all.values[0]

        return shap_values

    def _fallback_feature_importance(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Fallback method using model coefficients when SHAP fails.

        Args:
            features_df: Features DataFrame

        Returns:
            Array of importance scores
        """
        logger.debug("Using fallback feature importance (model coefficients)")

        # Scale features
        features_scaled = self.scaler.transform(features_df)

        # For logistic regression, use coefficients
        if hasattr(self.model, 'coef_'):
            # Average absolute coefficients across classes
            importance = np.abs(self.model.coef_).mean(axis=0)
            # Multiply by feature values for instance-specific importance
            importance = importance * np.abs(features_scaled[0])
        elif hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            importance = self.model.feature_importances_
        else:
            # Default: all equal importance
            importance = np.ones(features_df.shape[1])

        return importance

    def _get_top_features(
        self,
        feature_importance: Dict[str, float],
        feature_values: pd.Series,
        top_n: int
    ) -> List[Dict]:
        """
        Get top N features by absolute SHAP value with interpretations.

        Args:
            feature_importance: Dict of feature SHAP values
            feature_values: Series of feature values
            top_n: Number of top features

        Returns:
            List of dicts with feature info
        """
        # Sort by absolute SHAP value
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        top_features = []

        for feature_name, shap_value in sorted_features:
            feature_value = feature_values[feature_name]

            # Get feature metadata from config
            feature_info = self.feature_explanations.get(feature_name, {})

            # Determine if feature increases or decreases risk
            impact = "INCREASES" if shap_value > 0 else "DECREASES"

            # Get healthy range
            healthy_range = HEALTHY_RANGES.get(feature_name, (None, None))

            # Build explanation
            explanation = self._format_feature_explanation(
                feature_name,
                feature_value,
                shap_value,
                impact,
                healthy_range,
                feature_info
            )

            top_features.append({
                'feature_name': feature_name,
                'feature_display_name': feature_info.get('name', feature_name),
                'value': float(feature_value),
                'shap_value': float(shap_value),
                'abs_shap_value': abs(float(shap_value)),
                'impact': impact,
                'healthy_range': healthy_range,
                'explanation': explanation,
                'unit': feature_info.get('unit', '')
            })

        return top_features

    def _format_feature_explanation(
        self,
        feature_name: str,
        value: float,
        shap_value: float,
        impact: str,
        healthy_range: Tuple[float, float],
        feature_info: Dict
    ) -> str:
        """
        Format a human-readable explanation for a feature.

        Args:
            feature_name: Feature name
            value: Feature value
            shap_value: SHAP value
            impact: "INCREASES" or "DECREASES"
            healthy_range: Tuple of (min, max) healthy range
            feature_info: Feature metadata from config

        Returns:
            Formatted explanation string
        """
        display_name = feature_info.get('name', feature_name)
        unit = feature_info.get('unit', '')

        # Format value with unit
        if unit:
            value_str = f"{value:.1f} {unit}"
        else:
            value_str = f"{value:.2f}"

        # Build explanation
        explanation = f"{display_name} ({value_str}) - {impact} risk by {abs(shap_value):.2f}"

        # Add healthy range context
        if healthy_range[0] is not None and healthy_range[1] is not None:
            min_val, max_val = healthy_range

            if value < min_val:
                explanation += f"\n  → Below healthy range: {min_val}-{max_val} {unit}"
            elif value > max_val:
                explanation += f"\n  → Above healthy range: {min_val}-{max_val} {unit}"
            else:
                explanation += f"\n  → Within healthy range: {min_val}-{max_val} {unit}"

        # Add interpretation from config
        interpretation = feature_info.get('interpretation', {})
        if impact == "INCREASES" and 'high' in interpretation:
            explanation += f"\n  → {interpretation['high']}"
        elif impact == "DECREASES" and 'low' in interpretation:
            explanation += f"\n  → {interpretation['low']}"

        return explanation

    def _generate_summary(
        self,
        top_features: List[Dict],
        feature_values: pd.Series
    ) -> str:
        """
        Generate human-readable summary of explanation.

        Args:
            top_features: List of top contributing features
            feature_values: Series of all feature values

        Returns:
            Summary string
        """
        summary_parts = [
            "Top Contributing Factors:\n"
        ]

        for i, feature in enumerate(top_features, 1):
            summary_parts.append(
                f"{i}. {feature['explanation']}\n"
            )

        return "\n".join(summary_parts)


def explain_prediction(
    features: pd.DataFrame,
    patient_id: Optional[int] = None,
    model_path: str = None,
    scaler_path: str = None
) -> Dict:
    """
    Convenience function to explain a single prediction.

    Args:
        features: Features DataFrame or Series
        patient_id: Patient identifier
        model_path: Optional model path
        scaler_path: Optional scaler path

    Returns:
        Explanation dictionary

    Example:
        >>> explanation = explain_prediction(features, patient_id=123)
        >>> print(explanation['summary'])
    """
    explainer = RiskExplainer(model_path, scaler_path)
    return explainer.explain(features, patient_id)


# Example usage
if __name__ == "__main__":
    from src.data.ingestion import load_csv
    from src.data.preprocessing import clean_data
    from src.data.feature_engineering import (
        create_cardiovascular_features,
        get_feature_columns
    )
    from src.ml.predictor import RiskPredictor

    # Load sample data
    print("Loading sample data...")
    df = load_csv(limit=100)
    df_clean = clean_data(df)
    features_df = create_cardiovascular_features(df_clean)

    feature_cols = get_feature_columns()
    X = features_df[feature_cols].dropna()

    if len(X) == 0:
        print("No valid features found")
        exit()

    try:
        # Initialize predictor and explainer
        predictor = RiskPredictor()
        explainer = RiskExplainer()

        # Get prediction and explanation for first patient
        patient_features = X.iloc[0]
        patient_id = features_df.iloc[0]['user_id'] if 'user_id' in features_df.columns else 1

        print(f"\n=== Risk Prediction and Explanation for Patient {patient_id} ===\n")

        # Predict
        prediction = predictor.predict(patient_features)
        print(f"Risk Score: {prediction['risk_score']:.3f}")
        print(f"Predicted Risk Level: {prediction['predicted_label']}")
        print(f"\nProbabilities:")
        for label, prob in prediction['risk_probabilities'].items():
            print(f"  {label}: {prob:.3f}")

        # Explain
        print(f"\n{'-'*60}")
        explanation = explainer.explain(patient_features, patient_id)
        print(explanation['summary'])

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Model must be trained first. Run trainer.py to train a model.")
