"""
CardioGuard Configuration Settings

Centralized configuration management for all application settings.
Reads from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    """Application configuration settings"""

    # FHIR Server Configuration
    FHIR_SERVER_URL = os.getenv("FHIR_SERVER_URL", "http://localhost:8080/fhir")
    FHIR_VERSION = "R4"

    # Data Paths
    DATASET_PATH = os.getenv(
        "DATASET_PATH",
        str(BASE_DIR / "data" / "raw" / "fitness_tracker_dataset.csv")
    )
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "models" / "risk_classifier.pkl")
    )
    SCALER_PATH = os.getenv(
        "SCALER_PATH",
        str(BASE_DIR / "models" / "feature_scaler.pkl")
    )
    SQLITE_DB_PATH = os.getenv(
        "SQLITE_DB_PATH",
        str(BASE_DIR / "data" / "cache" / "cardioguard.db")
    )

    # Data Configuration
    DATA_LIMIT = int(os.getenv("DATA_LIMIT", 10000))
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

    # ML Configuration
    ML_TEST_SIZE = float(os.getenv("ML_TEST_SIZE", 0.2))
    ML_MODEL_TYPE = os.getenv("ML_MODEL_TYPE", "logistic")  # 'logistic' or 'decision_tree'
    ML_MAX_ITER = int(os.getenv("ML_MAX_ITER", 1000))

    # Risk Stratification Thresholds
    RISK_THRESHOLD_LOW = float(os.getenv("RISK_THRESHOLD_LOW", 0.33))
    RISK_THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", 0.66))

    # FHIR Client Configuration
    FHIR_BATCH_SIZE = int(os.getenv("FHIR_BATCH_SIZE", 100))
    FHIR_RETRY_ATTEMPTS = int(os.getenv("FHIR_RETRY_ATTEMPTS", 3))
    FHIR_RETRY_BACKOFF = int(os.getenv("FHIR_RETRY_BACKOFF", 2))
    FHIR_TIMEOUT = int(os.getenv("FHIR_TIMEOUT", 30))

    # Feature Engineering Configuration
    RESTING_HR_STEPS_THRESHOLD = int(os.getenv("RESTING_HR_STEPS_THRESHOLD", 1000))
    SEDENTARY_STEPS_THRESHOLD = int(os.getenv("SEDENTARY_STEPS_THRESHOLD", 5000))
    ROLLING_WINDOW_SHORT = int(os.getenv("ROLLING_WINDOW_SHORT", 7))
    ROLLING_WINDOW_LONG = int(os.getenv("ROLLING_WINDOW_LONG", 30))

    # Authentication (Demo - DO NOT use in production)
    DEMO_USERS = {
        os.getenv("DEMO_USERNAME_1", "clinician1"): os.getenv("DEMO_PASSWORD_1", "demo123"),
        os.getenv("DEMO_USERNAME_2", "admin"): os.getenv("DEMO_PASSWORD_2", "admin456")
    }

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Streamlit Configuration
    STREAMLIT_PAGE_TITLE = "CardioGuard - Cardiovascular Wellness Monitoring"
    STREAMLIT_PAGE_ICON = "❤️"
    STREAMLIT_LAYOUT = "wide"

    # Risk Thresholds Configuration File
    RISK_THRESHOLDS_CONFIG = str(BASE_DIR / "config" / "risk_thresholds.yaml")

    # SHAP Explainer Configuration
    SHAP_TOP_FEATURES = int(os.getenv("SHAP_TOP_FEATURES", 3))

    # Educational Disclaimer
    DISCLAIMER_TEXT = (
        "⚠️ EDUCATIONAL DEMONSTRATION ONLY\n\n"
        "This application does NOT provide medical advice, diagnosis, or treatment.\n"
        "For medical concerns, consult a qualified healthcare provider."
    )

    @classmethod
    def validate(cls):
        """Validate critical configuration settings"""
        errors = []

        # Check if dataset exists
        if not Path(cls.DATASET_PATH).exists():
            errors.append(f"Dataset not found at: {cls.DATASET_PATH}")

        # Check if risk thresholds config exists
        if not Path(cls.RISK_THRESHOLDS_CONFIG).exists():
            errors.append(f"Risk thresholds config not found at: {cls.RISK_THRESHOLDS_CONFIG}")

        # Validate threshold ranges
        if not (0 < cls.RISK_THRESHOLD_LOW < cls.RISK_THRESHOLD_HIGH < 1):
            errors.append(
                f"Invalid risk thresholds: LOW={cls.RISK_THRESHOLD_LOW}, "
                f"HIGH={cls.RISK_THRESHOLD_HIGH}. Must be 0 < LOW < HIGH < 1"
            )

        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))

        return True


# Create singleton instance
settings = Settings()


# Validate settings on import (can be disabled in tests)
if os.getenv("SKIP_CONFIG_VALIDATION") != "true":
    try:
        settings.validate()
    except ValueError as e:
        # Print warning but don't crash on import
        # This allows for gradual setup during development
        print(f"⚠️  Configuration warning: {e}")
