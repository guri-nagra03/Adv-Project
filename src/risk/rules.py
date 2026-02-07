"""
Rule-Based Risk Overrides

Applies deterministic rules to override ML predictions in edge cases.

Override scenarios:
- Extreme sedentary behavior (steps < 2000/day)
- Severe sleep deprivation (< 5 hours/night)
- Tachycardia pattern (resting HR > 100 bpm)

These rules ensure critical risk indicators are flagged even if ML
score is low.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.utils.constants import RISK_LEVEL_RED
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class RuleEvaluationError(Exception):
    """Raised when rule evaluation fails"""
    pass


def check_override_rules(
    features: Dict[str, float],
    override_rules: Dict[str, Dict]
) -> Dict:
    """
    Check if any override rules are triggered.

    Args:
        features: Patient features dictionary
        override_rules: Override rules from config

    Returns:
        Dictionary with:
        - override_triggered: Boolean
        - triggered_rule: Rule name if triggered
        - force_level: Risk level to force
        - reason: Human-readable reason

    Example:
        >>> features = {'resting_hr_estimate': 105, 'steps_avg_30d': 8000}
        >>> override_rules = {
        ...     'tachycardia_pattern': {
        ...         'condition': 'resting_hr_estimate > 100',
        ...         'force_level': 'red',
        ...         'reason': 'Elevated resting heart rate'
        ...     }
        ... }
        >>> result = check_override_rules(features, override_rules)
        >>> result['override_triggered']  # True
        >>> result['triggered_rule']  # 'tachycardia_pattern'
    """
    for rule_name, rule_config in override_rules.items():
        condition = rule_config.get('condition')
        force_level = rule_config.get('force_level', RISK_LEVEL_RED)
        reason = rule_config.get('reason', f'Rule triggered: {rule_name}')

        # Evaluate condition
        if evaluate_condition(condition, features):
            logger.info(
                f"Override rule triggered: {rule_name} - {reason}"
            )

            return {
                'override_triggered': True,
                'triggered_rule': rule_name,
                'force_level': force_level.capitalize(),  # Normalize to 'Red'
                'reason': reason
            }

    # No rules triggered
    return {
        'override_triggered': False,
        'triggered_rule': None,
        'force_level': None,
        'reason': None
    }


def evaluate_condition(condition: str, features: Dict[str, float]) -> bool:
    """
    Evaluate a rule condition against patient features.

    Supports simple comparisons:
    - 'feature_name > value'
    - 'feature_name < value'
    - 'feature_name >= value'
    - 'feature_name <= value'
    - 'feature_name == value'

    Args:
        condition: Condition string (e.g., 'resting_hr_estimate > 100')
        features: Patient features dictionary

    Returns:
        True if condition is met, False otherwise

    Raises:
        RuleEvaluationError: If condition format is invalid

    Example:
        >>> features = {'resting_hr_estimate': 95}
        >>> evaluate_condition('resting_hr_estimate > 90', features)  # True
        >>> evaluate_condition('resting_hr_estimate < 90', features)  # False
    """
    try:
        # Parse condition
        condition = condition.strip()

        # Determine operator
        for op in ['>=', '<=', '==', '>', '<']:
            if op in condition:
                parts = condition.split(op)
                if len(parts) != 2:
                    raise RuleEvaluationError(
                        f"Invalid condition format: {condition}"
                    )

                feature_name = parts[0].strip()
                threshold_str = parts[1].strip()

                # Get feature value
                if feature_name not in features:
                    logger.warning(
                        f"Feature '{feature_name}' not found in features, "
                        f"assuming condition is False"
                    )
                    return False

                feature_value = features[feature_name]

                # Handle missing values
                if pd.isna(feature_value):
                    logger.debug(
                        f"Feature '{feature_name}' is NaN, condition is False"
                    )
                    return False

                # Parse threshold
                try:
                    threshold = float(threshold_str)
                except ValueError:
                    raise RuleEvaluationError(
                        f"Invalid threshold value: {threshold_str}"
                    )

                # Evaluate comparison
                if op == '>':
                    result = feature_value > threshold
                elif op == '<':
                    result = feature_value < threshold
                elif op == '>=':
                    result = feature_value >= threshold
                elif op == '<=':
                    result = feature_value <= threshold
                elif op == '==':
                    result = abs(feature_value - threshold) < 1e-6

                logger.debug(
                    f"Condition '{condition}': "
                    f"{feature_value} {op} {threshold} = {result}"
                )

                return result

        # No operator found
        raise RuleEvaluationError(
            f"No valid operator found in condition: {condition}"
        )

    except Exception as e:
        logger.error(f"Error evaluating condition '{condition}': {e}")
        raise RuleEvaluationError(f"Failed to evaluate condition: {e}")


def apply_override_rules(
    ml_level: str,
    features: Dict[str, float],
    override_rules: Dict[str, Dict]
) -> Tuple[str, Optional[str]]:
    """
    Apply override rules to ML-based risk level.

    Args:
        ml_level: ML-predicted risk level ('Green', 'Yellow', 'Red')
        features: Patient features
        override_rules: Override rules from config

    Returns:
        Tuple of (final_level, override_reason)
        If no override, returns (ml_level, None)

    Example:
        >>> ml_level = 'Yellow'
        >>> features = {'steps_avg_30d': 1500}  # Very low activity
        >>> override_rules = {
        ...     'extreme_sedentary': {
        ...         'condition': 'steps_avg_30d < 2000',
        ...         'force_level': 'red'
        ...     }
        ... }
        >>> final_level, reason = apply_override_rules(ml_level, features, override_rules)
        >>> final_level  # 'Red'
        >>> reason  # 'extreme_sedentary'
    """
    result = check_override_rules(features, override_rules)

    if result['override_triggered']:
        return result['force_level'], result['triggered_rule']
    else:
        return ml_level, None


def get_triggered_rules(
    features: Dict[str, float],
    override_rules: Dict[str, Dict]
) -> List[Dict]:
    """
    Get all triggered rules (for debugging/explanation).

    Args:
        features: Patient features
        override_rules: Override rules from config

    Returns:
        List of triggered rule details

    Example:
        >>> features = {'resting_hr_estimate': 105, 'steps_avg_30d': 1500}
        >>> triggered = get_triggered_rules(features, override_rules)
        >>> len(triggered)  # 2 (both tachycardia and sedentary)
    """
    triggered = []

    for rule_name, rule_config in override_rules.items():
        condition = rule_config.get('condition')

        if evaluate_condition(condition, features):
            triggered.append({
                'rule_name': rule_name,
                'condition': condition,
                'force_level': rule_config.get('force_level', RISK_LEVEL_RED),
                'reason': rule_config.get('reason', '')
            })

    return triggered


# Example usage
if __name__ == "__main__":
    import yaml
    from config.settings import settings

    print("=== Rule-Based Override Demo ===\n")

    # Load override rules from config
    with open(settings.RISK_THRESHOLDS_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    override_rules = config.get('override_rules', {})

    print(f"Loaded {len(override_rules)} override rules:")
    for rule_name, rule_config in override_rules.items():
        print(f"  - {rule_name}: {rule_config.get('condition')}")

    # Test case 1: Extreme sedentary
    print("\n" + "=" * 60)
    print("Test 1: Extreme Sedentary Patient")
    print("-" * 60)

    features_sedentary = {
        'resting_hr_estimate': 75,
        'sleep_hours_avg': 7.2,
        'sleep_hours_avg_7d': 7.2,
        'steps_avg_30d': 1500,  # Very low!
        'activity_score': 15,
        'sedentary_ratio': 0.85
    }

    result = check_override_rules(features_sedentary, override_rules)
    print(f"Features: steps_avg_30d = {features_sedentary['steps_avg_30d']}")
    print(f"Override triggered: {result['override_triggered']}")
    if result['override_triggered']:
        print(f"Rule: {result['triggered_rule']}")
        print(f"Force level: {result['force_level']}")
        print(f"Reason: {result['reason']}")

    # Test case 2: Sleep deprivation
    print("\n" + "=" * 60)
    print("Test 2: Severe Sleep Deprivation")
    print("-" * 60)

    features_sleep = {
        'resting_hr_estimate': 80,
        'sleep_hours_avg': 4.5,
        'sleep_hours_avg_7d': 4.5,  # Very low!
        'steps_avg_30d': 8000,
        'activity_score': 45,
        'sedentary_ratio': 0.3
    }

    result = check_override_rules(features_sleep, override_rules)
    print(f"Features: sleep_hours_avg_7d = {features_sleep['sleep_hours_avg_7d']}")
    print(f"Override triggered: {result['override_triggered']}")
    if result['override_triggered']:
        print(f"Rule: {result['triggered_rule']}")
        print(f"Force level: {result['force_level']}")
        print(f"Reason: {result['reason']}")

    # Test case 3: Tachycardia
    print("\n" + "=" * 60)
    print("Test 3: Tachycardia Pattern")
    print("-" * 60)

    features_hr = {
        'resting_hr_estimate': 105,  # Very high!
        'sleep_hours_avg': 7.5,
        'sleep_hours_avg_7d': 7.5,
        'steps_avg_30d': 10000,
        'activity_score': 60,
        'sedentary_ratio': 0.2
    }

    result = check_override_rules(features_hr, override_rules)
    print(f"Features: resting_hr_estimate = {features_hr['resting_hr_estimate']}")
    print(f"Override triggered: {result['override_triggered']}")
    if result['override_triggered']:
        print(f"Rule: {result['triggered_rule']}")
        print(f"Force level: {result['force_level']}")
        print(f"Reason: {result['reason']}")

    # Test case 4: Healthy patient (no overrides)
    print("\n" + "=" * 60)
    print("Test 4: Healthy Patient (No Overrides)")
    print("-" * 60)

    features_healthy = {
        'resting_hr_estimate': 68,
        'sleep_hours_avg': 7.8,
        'sleep_hours_avg_7d': 7.8,
        'steps_avg_30d': 12000,
        'activity_score': 75,
        'sedentary_ratio': 0.15
    }

    result = check_override_rules(features_healthy, override_rules)
    print(f"Features: All within healthy ranges")
    print(f"Override triggered: {result['override_triggered']}")

    # Test case 5: Multiple rules triggered
    print("\n" + "=" * 60)
    print("Test 5: Multiple Rules Triggered")
    print("-" * 60)

    features_multiple = {
        'resting_hr_estimate': 105,  # High!
        'sleep_hours_avg': 4.2,
        'sleep_hours_avg_7d': 4.2,  # Low!
        'steps_avg_30d': 1200,  # Very low!
        'activity_score': 10,
        'sedentary_ratio': 0.9
    }

    all_triggered = get_triggered_rules(features_multiple, override_rules)
    print(f"Number of rules triggered: {len(all_triggered)}")
    for triggered in all_triggered:
        print(f"  - {triggered['rule_name']}: {triggered['reason']}")
