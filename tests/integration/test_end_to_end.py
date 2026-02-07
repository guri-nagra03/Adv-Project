"""
Integration tests for end-to-end workflows.
"""

import pytest
import pandas as pd

from src.data.ingestion import load_csv
from src.data.preprocessing import clean_data
from src.data.feature_engineering import create_cardiovascular_features, get_feature_columns
from src.ml.label_generator import generate_synthetic_labels
from src.ml.trainer import train_model
from src.ml.predictor import RiskPredictor
from src.risk.stratification import RiskStratifier
from src.fhir.converter import create_observation
from src.storage.sqlite_cache import SQLiteCache


class TestDataPipeline:
    """Integration tests for data pipeline."""

    def test_full_data_pipeline(self, sample_raw_data):
        """Test complete data pipeline from raw to features."""
        # Clean data
        df_clean = clean_data(sample_raw_data)
        assert len(df_clean) > 0

        # Create features
        features_df = create_cardiovascular_features(df_clean)
        assert len(features_df) > 0

        # Get feature columns
        feature_cols = get_feature_columns()
        assert all(col in features_df.columns for col in feature_cols)

        # Check for complete rows
        X = features_df[feature_cols].dropna()
        assert len(X) > 0


class TestMLPipeline:
    """Integration tests for ML pipeline."""

    def test_full_ml_pipeline(self, sample_features, sample_labels):
        """Test complete ML pipeline from features to predictions."""
        feature_cols = get_feature_columns()
        X = sample_features[feature_cols]

        # Train model
        model, scaler, metrics = train_model(X, sample_labels)

        assert model is not None
        assert scaler is not None
        assert 'test_accuracy' in metrics
        assert metrics['test_accuracy'] > 0

    def test_prediction_workflow(self, sample_features):
        """Test prediction workflow."""
        # Generate labels
        labels = generate_synthetic_labels(sample_features)
        assert len(labels) == len(sample_features)

        feature_cols = get_feature_columns()
        X = sample_features[feature_cols]

        # Train
        model, scaler, _ = train_model(X, labels)

        # Create predictor
        predictor = RiskPredictor(model=model, scaler=scaler)

        # Predict
        prediction = predictor.predict(X.iloc[0])

        assert 'risk_score' in prediction
        assert 'predicted_label' in prediction
        assert 'risk_probabilities' in prediction
        assert 0 <= prediction['risk_score'] <= 1


class TestRiskWorkflow:
    """Integration tests for risk stratification workflow."""

    def test_prediction_to_stratification(self, sample_features):
        """Test workflow from prediction to stratification."""
        # Generate labels and train
        labels = generate_synthetic_labels(sample_features)
        feature_cols = get_feature_columns()
        X = sample_features[feature_cols]
        model, scaler, _ = train_model(X, labels)

        # Predict
        predictor = RiskPredictor(model=model, scaler=scaler)
        prediction = predictor.predict(X.iloc[0])

        # Stratify
        stratifier = RiskStratifier()
        stratification = stratifier.stratify(
            ml_score=prediction['risk_score'],
            features=X.iloc[0].to_dict(),
            patient_id=1
        )

        assert 'risk_level' in stratification
        assert stratification['risk_level'] in ['Green', 'Yellow', 'Red']
        assert 'recommendations' in stratification
        assert len(stratification['recommendations']) > 0


class TestFHIRWorkflow:
    """Integration tests for FHIR workflow."""

    def test_data_to_fhir_observations(self, sample_raw_data):
        """Test converting data to FHIR Observations."""
        row = sample_raw_data.iloc[0]

        obs = create_observation(
            user_id=int(row['user_id']),
            date=str(row['date']),
            metric_name='steps',
            value=float(row['steps'])
        )

        # Validate observation structure
        assert obs.resourceType == 'Observation'
        assert obs.status == 'final'
        assert obs.code.coding[0].system == 'http://loinc.org'
        assert obs.subject.reference == f"Patient/{int(row['user_id'])}"
        assert obs.valueQuantity.value == float(row['steps'])


class TestStorageWorkflow:
    """Integration tests for storage workflow."""

    def test_cache_workflow(self, temp_db_path):
        """Test complete cache workflow."""
        cache = SQLiteCache(db_path=temp_db_path)

        # Create patient
        cache.upsert_patient(
            patient_id=123,
            latest_risk_level='Yellow',
            latest_ml_score=0.55
        )

        # Retrieve patient
        patient = cache.get_patient(123)
        assert patient is not None
        assert patient['patient_id'] == 123
        assert patient['latest_risk_level'] == 'Yellow'

        # Save prediction
        pred_id = cache.save_prediction(
            patient_id=123,
            ml_score=0.55,
            predicted_label='Medium Risk',
            probabilities={'Low': 0.25, 'Medium': 0.50, 'High': 0.25}
        )

        assert pred_id is not None

        # Retrieve prediction
        pred = cache.get_latest_prediction(123)
        assert pred is not None
        assert pred['ml_score'] == 0.55

        # Save stratification
        strat_id = cache.save_stratification(
            patient_id=123,
            risk_level='Yellow',
            ml_score=0.55,
            threshold_based_level='Yellow',
            override_applied=False,
            recommendations=['Test recommendation']
        )

        assert strat_id is not None

        # Retrieve stratification
        strat = cache.get_latest_stratification(123)
        assert strat is not None
        assert strat['risk_level'] == 'Yellow'

        # Get statistics
        stats = cache.get_stats()
        assert stats['total_patients'] == 1
        assert stats['total_predictions'] == 1
        assert stats['total_stratifications'] == 1


class TestEndToEndPatientProcessing:
    """End-to-end integration test for complete patient processing."""

    def test_complete_patient_workflow(self, sample_raw_data, temp_db_path):
        """Test complete workflow from raw data to stored stratification."""
        # Step 1: Data processing
        df_clean = clean_data(sample_raw_data)
        features_df = create_cardiovascular_features(df_clean)

        feature_cols = get_feature_columns()
        X = features_df[feature_cols].dropna()

        if len(X) == 0:
            pytest.skip("No valid features generated")

        # Step 2: ML pipeline
        labels = generate_synthetic_labels(features_df.loc[X.index])
        model, scaler, _ = train_model(X, labels)

        # Step 3: Prediction
        predictor = RiskPredictor(model=model, scaler=scaler)
        patient_features = X.iloc[0]
        prediction = predictor.predict(patient_features)

        # Step 4: Stratification
        stratifier = RiskStratifier()
        stratification = stratifier.stratify(
            ml_score=prediction['risk_score'],
            features=patient_features.to_dict(),
            patient_id=1
        )

        # Step 5: Storage
        cache = SQLiteCache(db_path=temp_db_path)

        cache.upsert_patient(
            patient_id=1,
            latest_risk_level=stratification['risk_level'],
            latest_ml_score=prediction['risk_score']
        )

        cache.save_prediction(
            patient_id=1,
            ml_score=prediction['risk_score'],
            predicted_label=prediction['predicted_label'],
            probabilities=prediction['risk_probabilities']
        )

        cache.save_stratification(
            patient_id=1,
            risk_level=stratification['risk_level'],
            ml_score=stratification['ml_score'],
            threshold_based_level=stratification['threshold_based_level'],
            override_applied=stratification['override_applied'],
            recommendations=stratification['recommendations']
        )

        # Step 6: Verify stored data
        patient = cache.get_patient(1)
        assert patient is not None
        assert patient['latest_risk_level'] == stratification['risk_level']

        stored_pred = cache.get_latest_prediction(1)
        assert stored_pred is not None
        assert stored_pred['ml_score'] == prediction['risk_score']

        stored_strat = cache.get_latest_stratification(1)
        assert stored_strat is not None
        assert stored_strat['risk_level'] == stratification['risk_level']

        # Verify end-to-end consistency
        assert patient['latest_ml_score'] == stored_pred['ml_score']
        assert stored_pred['ml_score'] == stored_strat['ml_score']
        assert patient['latest_risk_level'] == stored_strat['risk_level']
