"""
Training Script

Complete end-to-end training pipeline for CardioGuard:
1. Load and preprocess fitness tracker data
2. Engineer cardiovascular features
3. Generate synthetic risk labels
4. Train ML model (logistic regression)
5. Generate predictions and stratifications for all patients
6. Populate SQLite cache
7. Optionally post to FHIR server

Usage:
    python scripts/train_model.py [--no-fhiro] [--limit N]

Arguments:
    --no-fhir: Skip FHIR server operations (cache-only mode)
    --limit N: Limit to N rows (default: 10000)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.settings import settings
from src.data.ingestion import load_csv
from src.data.preprocessing import clean_data
from src.data.feature_engineering import (
    create_cardiovascular_features,
    get_feature_columns
)
from src.ml.label_generator import generate_synthetic_labels
from src.ml.trainer import train_model, save_model
from src.ml.predictor import RiskPredictor
from src.ml.explainer import RiskExplainer
from src.risk.stratification import RiskStratifier
from src.fhir.converter import create_observation, batch_convert_observations
from src.fhir.risk_resources import create_risk_assessment, create_risk_flag
from src.storage.fhir_repository import FHIRRepository
from src.utils.logging_config import setup_logging
from src.utils.constants import LOINC_CODES

logger = setup_logging(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CardioGuard ML model and populate database"
    )

    parser.add_argument(
        "--no-fhir",
        action="store_true",
        help="Skip FHIR server operations (cache-only mode)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Limit number of rows to process (default: 10000)"
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (use existing model)"
    )

    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    print("=" * 80)
    print("CardioGuard Training Pipeline")
    print("=" * 80)
    print()

    # Configuration
    print("Configuration:")
    print(f"  Data limit: {args.limit} rows")
    print(f"  FHIR operations: {'Disabled' if args.no_fhir else 'Enabled'}")
    print(f"  Skip training: {'Yes' if args.skip_training else 'No'}")
    print()

    # =========================================================================
    # Step 1: Load and Preprocess Data
    # =========================================================================
    print("Step 1: Loading and preprocessing data...")
    print("-" * 80)

    try:
        df = load_csv(limit=args.limit)
        print(f"✓ Loaded {len(df)} rows from dataset")

        df_clean = clean_data(df)
        print(f"✓ Cleaned data: {len(df_clean)} rows remaining")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        print(f"✗ Error loading data: {e}")
        print("\nMake sure the dataset exists at:")
        print(f"  {settings.DATASET_PATH}")
        return 1

    # =========================================================================
    # Step 2: Feature Engineering
    # =========================================================================
    print("\nStep 2: Engineering cardiovascular features...")
    print("-" * 80)

    try:
        features_df = create_cardiovascular_features(df_clean)
        print(f"✓ Created features for {len(features_df)} patient-days")

        feature_cols = get_feature_columns()
        print(f"✓ Feature columns: {len(feature_cols)}")
        for col in feature_cols:
            print(f"    - {col}")

        # Drop rows with missing features
        X = features_df[feature_cols].dropna()
        print(f"✓ Features ready: {len(X)} complete rows")

        if len(X) == 0:
            print("✗ No valid features found. Check data quality.")
            return 1

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        print(f"✗ Error creating features: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # =========================================================================
    # Step 3: Generate Synthetic Labels
    # =========================================================================
    print("\nStep 3: Generating synthetic risk labels...")
    print("-" * 80)

    try:
        # Align features_df with X (same indices)
        features_aligned = features_df.loc[X.index]

        y = generate_synthetic_labels(features_aligned)
        print(f"✓ Generated {len(y)} synthetic labels")

        # Label distribution
        label_counts = y.value_counts().sort_index()
        print("  Label distribution:")
        print(f"    Low Risk (0):    {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(y)*100:.1f}%)")
        print(f"    Medium Risk (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(y)*100:.1f}%)")
        print(f"    High Risk (2):   {label_counts.get(2, 0)} ({label_counts.get(2, 0)/len(y)*100:.1f}%)")

    except Exception as e:
        logger.error(f"Label generation failed: {e}")
        print(f"✗ Error generating labels: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # =========================================================================
    # Step 4: Train ML Model
    # =========================================================================
    if not args.skip_training:
        print("\nStep 4: Training ML model...")
        print("-" * 80)

        try:
            model, scaler, metrics = train_model(X, y)
            print(f"✓ Model trained successfully")

            print("  Model Performance:")
            print(f"    Accuracy:  {metrics.get('accuracy', float('nan')):.3f}")
            print(f"    Precision: {metrics.get('weighted_precision', float('nan')):.3f}")
            print(f"    Recall:    {metrics.get('weighted_recall', float('nan')):.3f}")
            print(f"    F1 Score:  {metrics.get('weighted_f1', float('nan')):.3f}")

            if 'cv_accuracy' in metrics:
                print(f"    CV Accuracy: {metrics['cv_accuracy']:.3f}")


            # Save model
            model_path = settings.MODEL_PATH
            scaler_path = settings.SCALER_PATH

            save_model(model, scaler, model_path, scaler_path)
            print(f"✓ Model saved to {model_path}")
            print(f"✓ Scaler saved to {scaler_path}")

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            print(f"✗ Error training model: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\nStep 4: Skipping model training (using existing model)...")
        print("-" * 80)

    # =========================================================================
    # Step 5: Generate Predictions
    # =========================================================================
    print("\nStep 5: Generating predictions for all patients...")
    print("-" * 80)

    try:
        predictor = RiskPredictor()
        stratifier = RiskStratifier()

        # Get predictions
        predictions_df = predictor.predict_batch(X)
        print(f"✓ Generated {len(predictions_df)} predictions")

        # Risk score distribution
        import numpy as np
        print(f"  Risk score statistics:")
        print(f"    Mean: {np.mean(predictions_df['risk_score']):.3f}")
        print(f"    Median: {np.median(predictions_df['risk_score']):.3f}")
        print(f"    Min: {np.min(predictions_df['risk_score']):.3f}")
        print(f"    Max: {np.max(predictions_df['risk_score']):.3f}")

    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        print(f"✗ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # =========================================================================
    # Step 6: Initialize Storage
    # =========================================================================
    print("\nStep 6: Initializing storage layer...")
    print("-" * 80)

    try:
        repo = FHIRRepository(enable_fhir=not args.no_fhir)
        print(f"✓ Repository initialized")
        print(f"  FHIR server: {'Available' if repo.is_fhir_available() else 'Unavailable (cache-only mode)'}")

    except Exception as e:
        logger.error(f"Storage initialization failed: {e}")
        print(f"✗ Error initializing storage: {e}")
        return 1

    # =========================================================================
    # Step 7: Process Patients and Populate Database
    # =========================================================================
    print("\nStep 7: Processing patients and populating database...")
    print("-" * 80)

    try:
        # Get unique patient IDs
        features_aligned = features_df.loc[X.index]
        patient_ids = features_aligned['user_id'].unique()

        print(f"Processing {len(patient_ids)} unique patients...")

        successful_patients = 0
        failed_patients = 0

        for patient_id in tqdm(patient_ids, desc="Processing patients"):
            try:
                # Get patient data
                patient_data = features_aligned[features_aligned['user_id'] == patient_id]

                if len(patient_data) == 0:
                    continue

                # Get latest features for this patient
                latest_features = patient_data.iloc[-1]
                latest_date = latest_features['date']

                # Get all feature values for stratification (including additional features for override rules)
                from src.utils.constants import ADDITIONAL_FEATURES
                all_features_list = feature_cols + ADDITIONAL_FEATURES
                feature_values = {k: latest_features.get(k, None) for k in all_features_list if k in latest_features.index}

                # Predict (using only ML training features)
                prediction = predictor.predict(latest_features[feature_cols])

                # Stratify (pass all features including additional ones for override rules)
                stratification = stratifier.stratify(
                    ml_score=prediction['risk_score'],
                    features=feature_values,
                    patient_id=patient_id
                )

                # Create FHIR observations for this patient's latest day
                observations = []
                observation_metadata = []

                # Get raw data for this patient's latest date
                patient_raw_data = df_clean[
                    (df_clean['user_id'] == patient_id) &
                    (df_clean['date'] == latest_date)
                ]

                if len(patient_raw_data) > 0:
                    latest_raw = patient_raw_data.iloc[-1]

                    # Create observations for available metrics
                    for metric_name in LOINC_CODES.keys():
                        if metric_name in latest_raw and pd.notna(latest_raw[metric_name]):
                            try:
                                # Convert to float and validate
                                value = float(latest_raw[metric_name])

                                # Skip if value is NaN or infinite
                                if not np.isfinite(value):
                                    logger.debug(f"Skipping non-finite value for {metric_name}: {value}")
                                    continue

                                obs = create_observation(
                                    user_id=patient_id,
                                    date=latest_date,
                                    metric_name=metric_name,
                                    value=value
                                )
                                observations.append(obs)

                                observation_metadata.append({
                                    'metric_name': metric_name,
                                    'value': value,
                                    'unit': obs.valueQuantity.unit,
                                    'date': str(latest_date)
                                })
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Skipping invalid value for {metric_name}: {e}")
                                continue

                # Create RiskAssessment
                risk_assessment = create_risk_assessment(
                    user_id=patient_id,
                    ml_score=prediction['risk_score'],
                    risk_level=stratification['risk_level']
                )

                # Create Flag (if applicable)
                flag = create_risk_flag(
                    user_id=patient_id,
                    risk_level=stratification['risk_level'],
                    reason=f"Cardiovascular wellness risk: {stratification['risk_level']}"
                )

                # Process patient through repository
                result = repo.process_patient(
                    patient_id=patient_id,
                    observations=observations,
                    observation_metadata=observation_metadata,
                    prediction=prediction,
                    stratification=stratification,
                    risk_assessment=risk_assessment,
                    flag=flag
                )

                if len(result.get('errors', [])) == 0:
                    successful_patients += 1
                else:
                    logger.warning(f"Patient {patient_id} processed with errors: {result['errors']}")
                    failed_patients += 1

            except Exception as e:
                logger.error(f"Failed to process patient {patient_id}: {e}")
                failed_patients += 1
                continue

        print(f"\n✓ Processing complete:")
        print(f"    Successful: {successful_patients}")
        print(f"    Failed: {failed_patients}")

    except Exception as e:
        logger.error(f"Patient processing failed: {e}")
        print(f"✗ Error processing patients: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # =========================================================================
    # Step 8: Summary Statistics
    # =========================================================================
    print("\nStep 8: Summary statistics...")
    print("-" * 80)

    try:
        stats = repo.get_cache_stats()

        print("Database Statistics:")
        print(f"  Total patients: {stats.get('total_patients', 0)}")
        print(f"  Total observations: {stats.get('total_observations', 0)}")
        print(f"  Total predictions: {stats.get('total_predictions', 0)}")
        print(f"  Total stratifications: {stats.get('total_stratifications', 0)}")

        risk_dist = stats.get('risk_distribution', {})
        if risk_dist:
            print("\n  Risk Distribution:")
            print(f"    Green (Low):    {risk_dist.get('Green', 0)}")
            print(f"    Yellow (Medium): {risk_dist.get('Yellow', 0)}")
            print(f"    Red (High):     {risk_dist.get('Red', 0)}")

    except Exception as e:
        logger.warning(f"Failed to get statistics: {e}")

    # =========================================================================
    # Completion
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training Pipeline Complete!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Start the Streamlit UI:")
    print("     streamlit run ui/app.py")
    print()
    print("  2. Or start with Docker:")
    print("     docker-compose up")
    print()
    print("  3. Login with demo credentials:")
    print("     Username: clinician1")
    print("     Password: demo123")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
