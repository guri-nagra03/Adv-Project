"""
FHIR Repository

Unified interface for FHIR operations with automatic fallback to local cache.

Features:
- Wraps FHIRClient with SQLiteCache
- Graceful degradation when FHIR server unavailable
- Automatic caching of FHIR operations
- Unified API for storage operations
- Retry logic and error handling

Usage:
    >>> repo = FHIRRepository()
    >>> # Post observation (tries FHIR server, falls back to cache)
    >>> obs_id = repo.post_observation(observation, patient_id=123)
    >>> # Get patient data (from cache)
    >>> patient = repo.get_patient(123)
"""

from typing import Dict, List, Optional, Union
from datetime import datetime

import pandas as pd
from fhir.resources.observation import Observation
from fhir.resources.riskassessment import RiskAssessment
from fhir.resources.flag import Flag

from src.fhir.client import FHIRClient, FHIRClientError
from src.storage.sqlite_cache import SQLiteCache
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class RepositoryError(Exception):
    """Raised when repository operation fails"""
    pass


class FHIRRepository:
    """
    Unified repository for FHIR and local cache operations.

    Provides single interface for:
    - Posting FHIR resources (with cache fallback)
    - Querying patient data (from cache)
    - Saving predictions and stratifications
    - Managing patient records

    Example:
        >>> repo = FHIRRepository()
        >>> # Post observation to FHIR server + cache metadata
        >>> obs_id = repo.post_observation(observation, patient_id=123)
        >>> # Get patient stratification from cache
        >>> strat = repo.get_latest_stratification(patient_id=123)
    """

    def __init__(
        self,
        fhir_client: FHIRClient = None,
        cache: SQLiteCache = None,
        enable_fhir: bool = True
    ):
        """
        Initialize FHIR repository.

        Args:
            fhir_client: Optional FHIRClient instance
            cache: Optional SQLiteCache instance
            enable_fhir: Whether to enable FHIR operations (default: True)
        """
        self.fhir_client = fhir_client or FHIRClient()
        self.cache = cache or SQLiteCache()
        self.enable_fhir = enable_fhir

        # Check FHIR server connectivity
        self.fhir_available = False
        if self.enable_fhir:
            try:
                self.fhir_client.check_server_status()
                self.fhir_available = True
                logger.info("FHIR server available")
            except FHIRClientError as e:
                logger.warning(f"FHIR server unavailable: {e}")
                logger.info("Operating in cache-only mode")

        logger.info("FHIR repository initialized")

    # =========================================================================
    # FHIR Resource Operations
    # =========================================================================

    def post_observation(
        self,
        observation: Observation,
        patient_id: int,
        metric_name: str = None,
        value: float = None,
        unit: str = None,
        observation_date: str = None
    ) -> Optional[str]:
        """
        Post Observation to FHIR server and cache metadata.

        Args:
            observation: FHIR Observation resource
            patient_id: Patient identifier
            metric_name: Metric name (for cache)
            value: Metric value (for cache)
            unit: Unit (for cache)
            observation_date: Observation date (for cache)

        Returns:
            FHIR resource ID if posted successfully, None otherwise

        Example:
            >>> obs = create_observation(user_id=123, date='2023-01-01', metric_name='steps', value=10000)
            >>> obs_id = repo.post_observation(obs, patient_id=123, metric_name='steps', value=10000, unit='steps', observation_date='2023-01-01')
        """
        fhir_resource_id = None

        # Try posting to FHIR server
        if self.fhir_available:
            try:
                fhir_resource_id = self.fhir_client.post_observation(observation)
                logger.debug(f"Posted Observation/{fhir_resource_id} to FHIR server")
            except FHIRClientError as e:
                logger.warning(f"Failed to post observation to FHIR: {e}")
                logger.info("Saving to cache only")

        # Save metadata to cache
        if metric_name and value is not None and observation_date:
            try:
                self.cache.save_observation_metadata(
                    patient_id=patient_id,
                    metric_name=metric_name,
                    value=value,
                    unit=unit or '',
                    observation_date=observation_date,
                    fhir_resource_id=fhir_resource_id
                )
            except Exception as e:
                logger.error(f"Failed to cache observation metadata: {e}")

        # Ensure patient exists
        self.cache.upsert_patient(patient_id)

        return fhir_resource_id

    def post_risk_assessment(
        self,
        risk_assessment: RiskAssessment,
        patient_id: int
    ) -> Optional[str]:
        """
        Post RiskAssessment to FHIR server.

        Args:
            risk_assessment: FHIR RiskAssessment resource
            patient_id: Patient identifier

        Returns:
            FHIR resource ID if posted successfully, None otherwise
        """
        if not self.fhir_available:
            logger.debug("FHIR unavailable, skipping RiskAssessment post")
            return None

        try:
            resource_id = self.fhir_client.post_risk_assessment(risk_assessment)
            logger.debug(f"Posted RiskAssessment/{resource_id} to FHIR server")
            return resource_id
        except FHIRClientError as e:
            logger.warning(f"Failed to post RiskAssessment: {e}")
            return None

    def post_flag(
        self,
        flag: Flag,
        patient_id: int
    ) -> Optional[str]:
        """
        Post Flag to FHIR server.

        Args:
            flag: FHIR Flag resource
            patient_id: Patient identifier

        Returns:
            FHIR resource ID if posted successfully, None otherwise
        """
        if not self.fhir_available:
            logger.debug("FHIR unavailable, skipping Flag post")
            return None

        if flag is None:
            logger.debug(f"No flag to post for patient {patient_id} (low risk)")
            return None

        try:
            resource_id = self.fhir_client.post_flag(flag)
            logger.debug(f"Posted Flag/{resource_id} to FHIR server")
            return resource_id
        except FHIRClientError as e:
            logger.warning(f"Failed to post Flag: {e}")
            return None

    def batch_post_observations(
        self,
        observations: List[Observation],
        patient_ids: List[int],
        metadata: List[Dict] = None
    ) -> List[Optional[str]]:
        """
        Post multiple observations in batch.

        Args:
            observations: List of Observation resources
            patient_ids: List of patient IDs (parallel to observations)
            metadata: Optional list of metadata dicts for caching

        Returns:
            List of FHIR resource IDs (None for failed posts)

        Example:
            >>> obs_list = [obs1, obs2, obs3]
            >>> patient_ids = [123, 123, 124]
            >>> metadata = [
            ...     {'metric_name': 'steps', 'value': 10000, 'unit': 'steps', 'date': '2023-01-01'},
            ...     {'metric_name': 'heart_rate_avg', 'value': 75, 'unit': 'bpm', 'date': '2023-01-01'},
            ...     {'metric_name': 'steps', 'value': 8000, 'unit': 'steps', 'date': '2023-01-01'}
            ... ]
            >>> ids = repo.batch_post_observations(obs_list, patient_ids, metadata)
        """
        resource_ids = []

        for i, (obs, patient_id) in enumerate(zip(observations, patient_ids)):
            meta = metadata[i] if metadata and i < len(metadata) else {}

            resource_id = self.post_observation(
                observation=obs,
                patient_id=patient_id,
                metric_name=meta.get('metric_name'),
                value=meta.get('value'),
                unit=meta.get('unit'),
                observation_date=meta.get('date')
            )

            resource_ids.append(resource_id)

        logger.info(
            f"Batch posted {len(observations)} observations: "
            f"{sum(1 for x in resource_ids if x is not None)} successful"
        )

        return resource_ids

    # =========================================================================
    # Patient Operations (Cache-based)
    # =========================================================================

    def get_patient(self, patient_id: int) -> Optional[Dict]:
        """Get patient from cache."""
        return self.cache.get_patient(patient_id)

    def get_all_patients(self) -> pd.DataFrame:
        """Get all patients from cache."""
        return self.cache.get_all_patients()

    def upsert_patient(
        self,
        patient_id: int,
        latest_risk_level: str = None,
        latest_ml_score: float = None,
        metadata: Dict = None
    ):
        """Create or update patient in cache."""
        self.cache.upsert_patient(
            patient_id=patient_id,
            latest_risk_level=latest_risk_level,
            latest_ml_score=latest_ml_score,
            metadata=metadata
        )

    # =========================================================================
    # Prediction Operations (Cache-based)
    # =========================================================================

    def save_prediction(
        self,
        patient_id: int,
        ml_score: float,
        predicted_label: str,
        probabilities: Dict[str, float],
        feature_values: Dict[str, float] = None
    ) -> int:
        """Save prediction to cache."""
        return self.cache.save_prediction(
            patient_id=patient_id,
            ml_score=ml_score,
            predicted_label=predicted_label,
            probabilities=probabilities,
            feature_values=feature_values
        )

    def get_latest_prediction(self, patient_id: int) -> Optional[Dict]:
        """Get latest prediction from cache."""
        return self.cache.get_latest_prediction(patient_id)

    def get_prediction_history(
        self,
        patient_id: int,
        limit: int = 10
    ) -> pd.DataFrame:
        """Get prediction history from cache."""
        return self.cache.get_prediction_history(patient_id, limit)

    # =========================================================================
    # Stratification Operations (Cache-based)
    # =========================================================================

    def save_stratification(
        self,
        patient_id: int,
        risk_level: str,
        ml_score: float,
        threshold_based_level: str,
        override_applied: bool,
        override_reason: str = None,
        recommendations: List[str] = None,
        risk_metadata: Dict = None
    ) -> int:
        """Save stratification to cache."""
        return self.cache.save_stratification(
            patient_id=patient_id,
            risk_level=risk_level,
            ml_score=ml_score,
            threshold_based_level=threshold_based_level,
            override_applied=override_applied,
            override_reason=override_reason,
            recommendations=recommendations,
            risk_metadata=risk_metadata
        )

    def get_latest_stratification(self, patient_id: int) -> Optional[Dict]:
        """Get latest stratification from cache."""
        return self.cache.get_latest_stratification(patient_id)

    def get_stratification_history(
        self,
        patient_id: int,
        limit: int = 10
    ) -> pd.DataFrame:
        """Get stratification history from cache."""
        return self.cache.get_stratification_history(patient_id, limit)

    # =========================================================================
    # Observation Operations (Cache-based)
    # =========================================================================

    def get_patient_observations(
        self,
        patient_id: int,
        metric_name: str = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get observations from cache."""
        return self.cache.get_patient_observations(
            patient_id=patient_id,
            metric_name=metric_name,
            limit=limit
        )

    # =========================================================================
    # Complete Patient Workflow
    # =========================================================================

    def process_patient(
        self,
        patient_id: int,
        observations: List[Observation],
        observation_metadata: List[Dict],
        prediction: Dict,
        stratification: Dict,
        risk_assessment: RiskAssessment = None,
        flag: Flag = None
    ) -> Dict:
        """
        Complete workflow for processing a patient.

        Performs:
        1. Post observations to FHIR + cache
        2. Save prediction to cache
        3. Save stratification to cache
        4. Post RiskAssessment to FHIR
        5. Post Flag to FHIR
        6. Update patient record

        Args:
            patient_id: Patient identifier
            observations: List of FHIR Observation resources
            observation_metadata: List of metadata dicts for caching
            prediction: Prediction result dict
            stratification: Stratification result dict
            risk_assessment: Optional RiskAssessment resource
            flag: Optional Flag resource

        Returns:
            Dictionary with operation results

        Example:
            >>> result = repo.process_patient(
            ...     patient_id=123,
            ...     observations=obs_list,
            ...     observation_metadata=meta_list,
            ...     prediction=prediction_result,
            ...     stratification=stratification_result,
            ...     risk_assessment=risk_assessment_resource,
            ...     flag=flag_resource
            ... )
        """
        results = {
            'patient_id': patient_id,
            'observation_ids': [],
            'prediction_id': None,
            'stratification_id': None,
            'risk_assessment_id': None,
            'flag_id': None,
            'errors': []
        }

        # 1. Post observations
        try:
            obs_ids = self.batch_post_observations(
                observations=observations,
                patient_ids=[patient_id] * len(observations),
                metadata=observation_metadata
            )
            results['observation_ids'] = obs_ids
        except Exception as e:
            logger.error(f"Failed to post observations: {e}")
            results['errors'].append(f"Observations: {e}")

        # 2. Save prediction
        try:
            pred_id = self.save_prediction(
                patient_id=patient_id,
                ml_score=prediction['risk_score'],
                predicted_label=prediction['predicted_label'],
                probabilities=prediction['risk_probabilities'],
                feature_values=prediction.get('feature_values')
            )
            results['prediction_id'] = pred_id
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            results['errors'].append(f"Prediction: {e}")

        # 3. Save stratification
        try:
            strat_id = self.save_stratification(
                patient_id=patient_id,
                risk_level=stratification['risk_level'],
                ml_score=stratification['ml_score'],
                threshold_based_level=stratification['threshold_based_level'],
                override_applied=stratification['override_applied'],
                override_reason=stratification.get('override_reason'),
                recommendations=stratification.get('recommendations'),
                risk_metadata=stratification.get('risk_metadata')
            )
            results['stratification_id'] = strat_id
        except Exception as e:
            logger.error(f"Failed to save stratification: {e}")
            results['errors'].append(f"Stratification: {e}")

        # 4. Post RiskAssessment
        if risk_assessment:
            try:
                ra_id = self.post_risk_assessment(risk_assessment, patient_id)
                results['risk_assessment_id'] = ra_id
            except Exception as e:
                logger.error(f"Failed to post RiskAssessment: {e}")
                results['errors'].append(f"RiskAssessment: {e}")

        # 5. Post Flag
        if flag:
            try:
                flag_id = self.post_flag(flag, patient_id)
                results['flag_id'] = flag_id
            except Exception as e:
                logger.error(f"Failed to post Flag: {e}")
                results['errors'].append(f"Flag: {e}")

        # 6. Update patient record
        try:
            self.upsert_patient(
                patient_id=patient_id,
                latest_risk_level=stratification['risk_level'],
                latest_ml_score=prediction['risk_score']
            )
        except Exception as e:
            logger.error(f"Failed to update patient: {e}")
            results['errors'].append(f"Patient update: {e}")

        logger.info(
            f"Processed patient {patient_id}: "
            f"{len(results['observation_ids'])} observations, "
            f"risk level {stratification['risk_level']}"
        )

        return results

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.get_stats()

    def log_operation(
        self,
        operation: str,
        status: str,
        message: str = None,
        patient_id: int = None
    ):
        """Log operation to cache."""
        self.cache.log_operation(operation, status, message, patient_id)

    def is_fhir_available(self) -> bool:
        """Check if FHIR server is available."""
        return self.fhir_available


# Example usage
if __name__ == "__main__":
    from src.data.ingestion import load_csv
    from src.data.preprocessing import clean_data
    from src.data.feature_engineering import create_cardiovascular_features
    from src.fhir.converter import create_observation
    from src.ml.predictor import RiskPredictor
    from src.risk.stratification import RiskStratifier

    print("=== FHIR Repository Demo ===\n")

    # Initialize repository
    repo = FHIRRepository()

    print(f"FHIR server available: {repo.is_fhir_available()}\n")

    # Load sample data
    print("Loading sample data...")
    df = load_csv(limit=10)
    df_clean = clean_data(df)

    # Get first patient
    patient_id = int(df_clean.iloc[0]['user_id'])
    patient_date = df_clean.iloc[0]['date']

    print(f"\nProcessing patient {patient_id}...\n")

    # Create observations
    observations = []
    observation_metadata = []

    for metric in ['steps', 'heart_rate_avg', 'sleep_hours']:
        if metric in df_clean.columns:
            value = df_clean.iloc[0][metric]
            obs = create_observation(
                user_id=patient_id,
                date=patient_date,
                metric_name=metric,
                value=value
            )
            observations.append(obs)
            observation_metadata.append({
                'metric_name': metric,
                'value': value,
                'unit': obs.valueQuantity.unit,
                'date': patient_date
            })

    print(f"Created {len(observations)} observations")

    # Post observations
    print("\nPosting observations...")
    obs_ids = repo.batch_post_observations(
        observations=observations,
        patient_ids=[patient_id] * len(observations),
        metadata=observation_metadata
    )
    print(f"Posted {sum(1 for x in obs_ids if x)} observations to FHIR")

    # Save dummy prediction
    print("\nSaving prediction...")
    repo.save_prediction(
        patient_id=patient_id,
        ml_score=0.45,
        predicted_label="Medium Risk",
        probabilities={"Low": 0.35, "Medium": 0.45, "High": 0.20}
    )

    # Save dummy stratification
    print("Saving stratification...")
    repo.save_stratification(
        patient_id=patient_id,
        risk_level="Yellow",
        ml_score=0.45,
        threshold_based_level="Yellow",
        override_applied=False,
        recommendations=[
            "Increase physical activity to 150 minutes/week",
            "Improve sleep hygiene"
        ],
        risk_metadata={"color": "#ffc107", "icon": "⚠"}
    )

    # Get patient data
    print("\n" + "=" * 60)
    print("Patient Data from Cache:\n")

    patient = repo.get_patient(patient_id)
    print(f"Patient {patient_id}:")
    print(f"  Risk Level: {patient['latest_risk_level']}")
    print(f"  ML Score: {patient['latest_ml_score']:.3f}")
    print(f"  Observations: {patient['total_observations']}")

    prediction = repo.get_latest_prediction(patient_id)
    print(f"\nLatest Prediction:")
    print(f"  ML Score: {prediction['ml_score']:.3f}")
    print(f"  Label: {prediction['predicted_label']}")

    stratification = repo.get_latest_stratification(patient_id)
    print(f"\nLatest Stratification:")
    print(f"  Risk Level: {stratification['risk_level']}")
    print(f"  Recommendations:")
    for rec in stratification['recommendations']:
        print(f"    - {rec}")

    # Cache stats
    print("\n" + "=" * 60)
    print("Cache Statistics:\n")
    stats = repo.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
