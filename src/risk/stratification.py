"""
Risk Stratification Engine

Converts ML risk scores (0-1 continuous) into discrete risk categories
(Green/Yellow/Red) based on configurable thresholds.

Includes:
- ML score to risk level mapping
- Rule-based overrides for edge cases
- Personalized recommendations
- Risk level metadata (colors, descriptions)
"""

import yaml
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd

from config.settings import settings
from src.utils.constants import (
    RISK_LEVEL_GREEN,
    RISK_LEVEL_YELLOW,
    RISK_LEVEL_RED
)
from src.utils.logging_config import setup_logging
from src.risk.rules import check_override_rules

logger = setup_logging(__name__)


class StratificationError(Exception):
    """Raised when risk stratification fails"""
    pass


class RiskStratifier:
    """
    Stratify patients into risk categories based on ML scores and rules.

    Example:
        >>> stratifier = RiskStratifier()
        >>> result = stratifier.stratify(ml_score=0.72, features={'resting_hr_estimate': 95})
        >>> print(result['risk_level'])  # 'Red'
        >>> print(result['recommendations'])
    """

    def __init__(self, config_path: str = None):
        """
        Initialize risk stratifier.

        Args:
            config_path: Path to risk_thresholds.yaml
        """
        config_path = config_path or settings.RISK_THRESHOLDS_CONFIG

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract risk categories
        self.categories = self.config.get('risk_categories', {})
        self.override_rules = self.config.get('override_rules', {})

        # Extract thresholds
        self.threshold_low = self.categories['green']['ml_score_max']
        self.threshold_high = self.categories['red']['ml_score_min']

        logger.info(
            f"Risk stratifier initialized: "
            f"Green < {self.threshold_low}, "
            f"Yellow {self.threshold_low}-{self.threshold_high}, "
            f"Red >= {self.threshold_high}"
        )

    def stratify(
        self,
        ml_score: float,
        features: Dict[str, float] = None,
        patient_id: Optional[int] = None,
        top_features: List[Dict] = None
    ) -> Dict:
        """
        Stratify patient into risk category.

        Args:
            ml_score: ML risk score (0-1)
            features: Patient features for rule evaluation
            patient_id: Patient identifier
            top_features: Top contributing features from SHAP

        Returns:
            Dictionary with:
            - risk_level: 'Green', 'Yellow', or 'Red'
            - ml_score: Original ML score
            - threshold_based_level: Level before rule overrides
            - override_applied: Boolean
            - override_reason: Reason if override applied
            - recommendations: List of recommendations
            - risk_metadata: Color, description, priority
            - timestamp: Stratification timestamp

        Example:
            >>> result = stratifier.stratify(
            ...     ml_score=0.45,
            ...     features={'resting_hr_estimate': 95, 'sleep_hours_avg': 5.2}
            ... )
            >>> result['risk_level']  # 'Red' (overridden due to high HR)
            >>> result['override_reason']  # 'tachycardia_pattern'
        """
        # Validate ML score
        if not (0 <= ml_score <= 1):
            raise StratificationError(
                f"ML score must be between 0 and 1, got {ml_score}"
            )

        # Get threshold-based risk level
        threshold_level = self._get_threshold_level(ml_score)

        # Check for rule-based overrides
        override_applied = False
        override_reason = None
        final_level = threshold_level

        if features:
            override_result = check_override_rules(features, self.override_rules)
            if override_result['override_triggered']:
                override_applied = True
                override_reason = override_result['triggered_rule']
                final_level = override_result['force_level']

                logger.info(
                    f"Patient {patient_id}: Override applied "
                    f"({override_reason}) - {threshold_level} → {final_level}"
                )

        # Get recommendations
        recommendations = self._generate_recommendations(
            final_level,
            features,
            top_features
        )

        # Get risk metadata
        risk_metadata = self._get_risk_metadata(final_level)

        result = {
            'patient_id': patient_id,
            'risk_level': final_level,
            'ml_score': float(ml_score),
            'threshold_based_level': threshold_level,
            'override_applied': override_applied,
            'override_reason': override_reason,
            'recommendations': recommendations,
            'risk_metadata': risk_metadata,
            'timestamp': datetime.now().isoformat()
        }

        logger.debug(
            f"Stratified patient {patient_id}: "
            f"ML={ml_score:.3f} → {final_level}"
        )

        return result

    def _get_threshold_level(self, ml_score: float) -> str:
        """
        Get risk level based on ML score thresholds.

        Args:
            ml_score: ML risk score (0-1)

        Returns:
            Risk level ('Green', 'Yellow', or 'Red')
        """
        if ml_score < self.threshold_low:
            return RISK_LEVEL_GREEN
        elif ml_score < self.threshold_high:
            return RISK_LEVEL_YELLOW
        else:
            return RISK_LEVEL_RED

    def _generate_recommendations(
        self,
        risk_level: str,
        features: Dict[str, float] = None,
        top_features: List[Dict] = None
    ) -> List[str]:
        """
        Generate personalized recommendations based on risk level and features.

        Args:
            risk_level: Risk category
            features: Patient features
            top_features: Top contributing features from SHAP

        Returns:
            List of recommendation strings
        """
        # Base recommendations from config
        category_config = self.categories.get(risk_level.lower(), {})
        recommendations = category_config.get('recommendations', []).copy()

        # Add feature-specific recommendations
        if top_features:
            feature_recs = self._get_feature_recommendations(top_features)
            recommendations.extend(feature_recs)

        # Add urgency-based recommendations
        if risk_level == RISK_LEVEL_RED:
            recommendations.insert(
                0,
                "⚠️ Schedule wellness consultation with healthcare provider"
            )
        elif risk_level == RISK_LEVEL_YELLOW:
            recommendations.insert(
                0,
                "Consider discussing wellness goals with healthcare provider"
            )

        return recommendations

    def _get_feature_recommendations(
        self,
        top_features: List[Dict]
    ) -> List[str]:
        """
        Generate recommendations based on top contributing features.

        Args:
            top_features: List of top features from SHAP explainer

        Returns:
            List of feature-specific recommendations
        """
        recommendations = []

        feature_explanations = self.config.get('feature_explanations', {})

        for feature in top_features:
            feature_name = feature.get('feature_name')
            impact = feature.get('impact', 'INCREASES')

            if impact == 'INCREASES':
                # Feature is increasing risk
                feature_config = feature_explanations.get(feature_name, {})
                interpretation = feature_config.get('interpretation', {})

                if 'high' in interpretation:
                    # This feature has high value - get high interpretation
                    recommendations.append(interpretation['high'])
                elif 'low' in interpretation:
                    # This feature has low value - get low interpretation
                    recommendations.append(interpretation['low'])

        # Deduplicate and limit
        recommendations = list(dict.fromkeys(recommendations))[:3]

        return recommendations

    def _get_risk_metadata(self, risk_level: str) -> Dict:
        """
        Get metadata for risk level (color, description, priority).

        Args:
            risk_level: Risk category

        Returns:
            Metadata dictionary
        """
        category_config = self.categories.get(risk_level.lower(), {})

        return {
            'color': category_config.get('color', '#6c757d'),
            'description': category_config.get('description', ''),
            'priority': category_config.get('label', 'medium'),
            'icon': self._get_risk_icon(risk_level)
        }

    def _get_risk_icon(self, risk_level: str) -> str:
        """Get icon for risk level."""
        icons = {
            RISK_LEVEL_GREEN: '✓',
            RISK_LEVEL_YELLOW: '⚠',
            RISK_LEVEL_RED: '🔴'
        }
        return icons.get(risk_level, '?')

    def batch_stratify(
        self,
        predictions: pd.DataFrame,
        features: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Stratify multiple patients at once.

        Args:
            predictions: DataFrame with columns ['patient_id', 'ml_score']
            features: Optional DataFrame with patient features

        Returns:
            DataFrame with stratification results

        Example:
            >>> predictions = pd.DataFrame({
            ...     'patient_id': [1, 2, 3],
            ...     'ml_score': [0.25, 0.55, 0.85]
            ... })
            >>> results = stratifier.batch_stratify(predictions)
        """
        logger.info(f"Batch stratifying {len(predictions)} patients...")

        results = []

        for idx, row in predictions.iterrows():
            patient_id = row['patient_id']
            ml_score = row['ml_score']

            # Get features for this patient if available
            patient_features = None
            if features is not None and 'user_id' in features.columns:
                patient_mask = features['user_id'] == patient_id
                if patient_mask.any():
                    patient_features = features[patient_mask].iloc[0].to_dict()

            # Stratify
            result = self.stratify(
                ml_score=ml_score,
                features=patient_features,
                patient_id=patient_id
            )

            results.append(result)

        results_df = pd.DataFrame(results)

        logger.info(
            f"Batch stratification complete: "
            f"{(results_df['risk_level'] == RISK_LEVEL_GREEN).sum()} Green, "
            f"{(results_df['risk_level'] == RISK_LEVEL_YELLOW).sum()} Yellow, "
            f"{(results_df['risk_level'] == RISK_LEVEL_RED).sum()} Red"
        )

        return results_df


def stratify_risk(
    ml_score: float,
    features: Dict[str, float] = None,
    patient_id: Optional[int] = None,
    config_path: str = None
) -> Dict:
    """
    Convenience function to stratify a single patient.

    Args:
        ml_score: ML risk score (0-1)
        features: Patient features for rule evaluation
        patient_id: Patient identifier
        config_path: Optional config path

    Returns:
        Stratification result dictionary

    Example:
        >>> result = stratify_risk(ml_score=0.72, patient_id=123)
        >>> print(result['risk_level'])
    """
    stratifier = RiskStratifier(config_path)
    return stratifier.stratify(ml_score, features, patient_id)


# Example usage
if __name__ == "__main__":
    from src.data.ingestion import load_csv
    from src.data.preprocessing import clean_data
    from src.data.feature_engineering import (
        create_cardiovascular_features,
        get_feature_columns
    )
    from src.ml.predictor import RiskPredictor

    print("=== Risk Stratification Demo ===\n")

    # Load and prepare data
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
        # Initialize predictor and stratifier
        predictor = RiskPredictor()
        stratifier = RiskStratifier()

        # Get prediction for first patient
        patient_features = X.iloc[0]
        patient_id = features_df.iloc[0]['user_id'] if 'user_id' in features_df.columns else 1

        print(f"Patient {patient_id}:")
        print("-" * 60)

        # Predict
        prediction = predictor.predict(patient_features)
        ml_score = prediction['risk_score']

        print(f"ML Risk Score: {ml_score:.3f}")
        print(f"ML Predicted Label: {prediction['predicted_label']}\n")

        # Stratify
        patient_features_dict = patient_features.to_dict()
        stratification = stratifier.stratify(
            ml_score=ml_score,
            features=patient_features_dict,
            patient_id=patient_id
        )

        print(f"Risk Level: {stratification['risk_level']} {stratification['risk_metadata']['icon']}")
        print(f"Color: {stratification['risk_metadata']['color']}")
        print(f"Threshold-based level: {stratification['threshold_based_level']}")

        if stratification['override_applied']:
            print(f"⚠️ Override applied: {stratification['override_reason']}")

        print(f"\nRecommendations:")
        for i, rec in enumerate(stratification['recommendations'], 1):
            print(f"  {i}. {rec}")

        # Batch stratification demo
        print("\n" + "=" * 60)
        print("=== Batch Stratification ===\n")

        # Get predictions for first 10 patients
        X_batch = X.head(10)
        predictions_batch = predictor.batch_predict(X_batch)

        # Prepare predictions DataFrame
        predictions_df = pd.DataFrame({
            'patient_id': features_df.head(10)['user_id'].values if 'user_id' in features_df.columns else range(1, 11),
            'ml_score': predictions_batch['risk_scores']
        })

        # Batch stratify
        results_df = stratifier.batch_stratify(predictions_df, features_df.head(10))

        print(f"Stratified {len(results_df)} patients:")
        print(results_df[['patient_id', 'ml_score', 'risk_level', 'override_applied']].to_string(index=False))

        # Summary
        print(f"\nSummary:")
        print(f"  Green: {(results_df['risk_level'] == RISK_LEVEL_GREEN).sum()}")
        print(f"  Yellow: {(results_df['risk_level'] == RISK_LEVEL_YELLOW).sum()}")
        print(f"  Red: {(results_df['risk_level'] == RISK_LEVEL_RED).sum()}")
        print(f"  Overrides: {results_df['override_applied'].sum()}")

    except FileNotFoundError:
        print("Error: Model not found. Train a model first:")
        print("  python scripts/train_model.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
