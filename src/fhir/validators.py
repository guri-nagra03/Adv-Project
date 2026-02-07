"""
FHIR Resource Validators

Validates FHIR resources before sending to server.
Ensures standards compliance and catches errors early.
"""

from typing import List, Dict, Tuple

from fhir.resources.observation import Observation
from fhir.resources.riskassessment import RiskAssessment
from fhir.resources.flag import Flag

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class ValidationError(Exception):
    """Raised when FHIR resource validation fails"""
    pass


def validate_observation(observation: Observation) -> Tuple[bool, List[str]]:
    """
    Validate FHIR Observation resource.

    Args:
        observation: Observation to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Required fields
    if not observation.status:
        errors.append("Missing required field: status")

    if not observation.code:
        errors.append("Missing required field: code")
    elif not observation.code.coding:
        errors.append("Code must have at least one Coding")

    if not observation.subject:
        errors.append("Missing required field: subject")

    if not observation.effectiveDateTime:
        errors.append("Missing required field: effectiveDateTime")

    if not observation.valueQuantity:
        errors.append("Missing required field: valueQuantity")
    else:
        if observation.valueQuantity.value is None:
            errors.append("valueQuantity.value is required")

    # Validate LOINC coding
    if observation.code and observation.code.coding:
        loinc_coding = observation.code.coding[0]
        if loinc_coding.system != "http://loinc.org":
            errors.append(
                f"Expected LOINC system, got: {loinc_coding.system}"
            )

        if not loinc_coding.code:
            errors.append("LOINC coding missing code")

    # Validate patient reference
    if observation.subject:
        if not observation.subject.reference:
            errors.append("Subject reference is empty")
        elif not observation.subject.reference.startswith("Patient/"):
            errors.append(
                f"Subject must reference a Patient, got: {observation.subject.reference}"
            )

    is_valid = len(errors) == 0

    if is_valid:
        logger.debug("Observation validation passed")
    else:
        logger.warning(f"Observation validation failed: {errors}")

    return is_valid, errors


def validate_risk_assessment(risk_assessment: RiskAssessment) -> Tuple[bool, List[str]]:
    """
    Validate FHIR RiskAssessment resource.

    Args:
        risk_assessment: RiskAssessment to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Required fields
    if not risk_assessment.status:
        errors.append("Missing required field: status")

    if not risk_assessment.subject:
        errors.append("Missing required field: subject")

    if not risk_assessment.prediction:
        errors.append("Missing required field: prediction")
    elif len(risk_assessment.prediction) == 0:
        errors.append("Prediction array is empty")
    else:
        # Validate first prediction
        prediction = risk_assessment.prediction[0]

        if prediction.probabilityDecimal is None:
            errors.append("Prediction missing probabilityDecimal")
        else:
            # Validate probability range
            if not (0 <= prediction.probabilityDecimal <= 1):
                errors.append(
                    f"Probability must be between 0 and 1, got: {prediction.probabilityDecimal}"
                )

        if not prediction.outcome:
            errors.append("Prediction missing outcome")

    # Validate patient reference
    if risk_assessment.subject:
        if not risk_assessment.subject.reference:
            errors.append("Subject reference is empty")
        elif not risk_assessment.subject.reference.startswith("Patient/"):
            errors.append(
                f"Subject must reference a Patient, got: {risk_assessment.subject.reference}"
            )

    is_valid = len(errors) == 0

    if is_valid:
        logger.debug("RiskAssessment validation passed")
    else:
        logger.warning(f"RiskAssessment validation failed: {errors}")

    return is_valid, errors


def validate_flag(flag: Flag) -> Tuple[bool, List[str]]:
    """
    Validate FHIR Flag resource.

    Args:
        flag: Flag to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Required fields
    if not flag.status:
        errors.append("Missing required field: status")

    if not flag.code:
        errors.append("Missing required field: code")

    if not flag.subject:
        errors.append("Missing required field: subject")

    # Validate patient reference
    if flag.subject:
        if not flag.subject.reference:
            errors.append("Subject reference is empty")
        elif not flag.subject.reference.startswith("Patient/"):
            errors.append(
                f"Subject must reference a Patient, got: {flag.subject.reference}"
            )

    # Validate category
    if flag.category:
        for category in flag.category:
            if not category.coding:
                errors.append("Category must have at least one Coding")

    is_valid = len(errors) == 0

    if is_valid:
        logger.debug("Flag validation passed")
    else:
        logger.warning(f"Flag validation failed: {errors}")

    return is_valid, errors


def validate_batch(
    observations: List[Observation] = None,
    risk_assessments: List[RiskAssessment] = None,
    flags: List[Flag] = None
) -> Dict[str, List[Tuple[int, List[str]]]]:
    """
    Validate batches of FHIR resources.

    Args:
        observations: List of Observations
        risk_assessments: List of RiskAssessments
        flags: List of Flags

    Returns:
        Dictionary with validation results:
        {
            'observations': [(index, errors), ...],
            'risk_assessments': [(index, errors), ...],
            'flags': [(index, errors), ...]
        }
    """
    results = {
        'observations': [],
        'risk_assessments': [],
        'flags': []
    }

    # Validate observations
    if observations:
        for i, obs in enumerate(observations):
            is_valid, errors = validate_observation(obs)
            if not is_valid:
                results['observations'].append((i, errors))

    # Validate risk assessments
    if risk_assessments:
        for i, ra in enumerate(risk_assessments):
            is_valid, errors = validate_risk_assessment(ra)
            if not is_valid:
                results['risk_assessments'].append((i, errors))

    # Validate flags
    if flags:
        for i, flag in enumerate(flags):
            is_valid, errors = validate_flag(flag)
            if not is_valid:
                results['flags'].append((i, errors))

    # Log summary
    total_errors = (
        len(results['observations']) +
        len(results['risk_assessments']) +
        len(results['flags'])
    )

    if total_errors > 0:
        logger.warning(f"Batch validation found {total_errors} resources with errors")
    else:
        logger.info("Batch validation passed for all resources")

    return results


# Example usage
if __name__ == "__main__":
    from src.fhir.converter import create_observation
    from src.fhir.risk_resources import create_risk_assessment, create_risk_flag
    from src.utils.constants import RISK_LEVEL_RED

    print("=== Testing FHIR Resource Validation ===\n")

    # Test valid Observation
    print("1. Valid Observation:")
    obs = create_observation(
        user_id=123,
        date="2023-01-01",
        metric_name="steps",
        value=10000
    )
    is_valid, errors = validate_observation(obs)
    if is_valid:
        print("   ✓ Validation passed")
    else:
        print(f"   ✗ Validation failed: {errors}")

    # Test valid RiskAssessment
    print("\n2. Valid RiskAssessment:")
    risk = create_risk_assessment(
        user_id=123,
        ml_score=0.72,
        risk_level=RISK_LEVEL_RED
    )
    is_valid, errors = validate_risk_assessment(risk)
    if is_valid:
        print("   ✓ Validation passed")
    else:
        print(f"   ✗ Validation failed: {errors}")

    # Test valid Flag
    print("\n3. Valid Flag:")
    flag = create_risk_flag(
        user_id=123,
        risk_level=RISK_LEVEL_RED,
        reason="High risk"
    )
    is_valid, errors = validate_flag(flag)
    if is_valid:
        print("   ✓ Validation passed")
    else:
        print(f"   ✗ Validation failed: {errors}")

    # Test batch validation
    print("\n4. Batch Validation:")
    obs_list = [
        create_observation(123, "2023-01-01", "steps", 10000),
        create_observation(124, "2023-01-01", "heart_rate_avg", 75),
    ]
    results = validate_batch(observations=obs_list)
    if sum(len(v) for v in results.values()) == 0:
        print("   ✓ All resources valid")
    else:
        print(f"   ✗ Validation errors: {results}")
