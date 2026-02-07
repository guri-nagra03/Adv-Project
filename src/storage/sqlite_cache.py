"""
SQLite Cache

Local database for caching processed data, predictions, and risk stratifications.

Schema:
- patients: Patient metadata
- predictions: ML prediction results
- stratifications: Risk stratification results
- observations: Cached FHIR Observation metadata
- processing_log: Data processing history

Features:
- Fast local queries without hitting FHIR server
- Stores prediction/stratification history
- Tracks data processing status
- Provides audit trail
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd

from config.settings import settings
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class CacheError(Exception):
    """Raised when cache operation fails"""
    pass


class SQLiteCache:
    """
    SQLite-based local cache for CardioGuard data.

    Example:
        >>> cache = SQLiteCache()
        >>> cache.save_prediction(patient_id=123, ml_score=0.72, predicted_label='Red')
        >>> prediction = cache.get_latest_prediction(patient_id=123)
    """

    def __init__(self, db_path: str = None):
        """
        Initialize SQLite cache.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or settings.SQLITE_DB_PATH

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        logger.info(f"SQLite cache initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database schema if not exists."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id INTEGER PRIMARY KEY,
                first_seen_at TEXT NOT NULL,
                last_updated_at TEXT NOT NULL,
                total_observations INTEGER DEFAULT 0,
                latest_risk_level TEXT,
                latest_ml_score REAL,
                metadata TEXT
            )
        """)

        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                ml_score REAL NOT NULL,
                predicted_label TEXT NOT NULL,
                probabilities TEXT NOT NULL,
                feature_values TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)

        # Stratifications table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stratifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                ml_score REAL NOT NULL,
                threshold_based_level TEXT NOT NULL,
                override_applied INTEGER DEFAULT 0,
                override_reason TEXT,
                recommendations TEXT,
                risk_metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)

        # Observations table (metadata only, not full FHIR resources)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                fhir_resource_id TEXT,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                observation_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)

        # Processing log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                operation TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_patient_created
            ON predictions(patient_id, created_at DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_stratifications_patient_created
            ON stratifications(patient_id, created_at DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_patient_date
            ON observations(patient_id, observation_date DESC)
        """)

        conn.commit()
        conn.close()

        logger.debug("Database schema initialized")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper settings for concurrency."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=60.0,  # Wait up to 60 seconds for locks
            check_same_thread=False,
            isolation_level='DEFERRED'  # Use deferred transactions for better concurrency
        )
        conn.row_factory = sqlite3.Row  # Enable column access by name
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")  # 60 second busy timeout
        return conn

    # =========================================================================
    # Patient Operations
    # =========================================================================

    def upsert_patient(
        self,
        patient_id: int,
        latest_risk_level: str = None,
        latest_ml_score: float = None,
        metadata: Dict = None
    ):
        """
        Insert or update patient record.

        Args:
            patient_id: Patient identifier
            latest_risk_level: Latest risk level
            latest_ml_score: Latest ML score
            metadata: Additional metadata
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        # Ensure proper types
        patient_id = int(patient_id)

        # Check if patient exists
        cursor.execute(
            "SELECT patient_id FROM patients WHERE patient_id = ?",
            (patient_id,)
        )
        exists = cursor.fetchone() is not None

        if exists:
            # Update existing
            cursor.execute("""
                UPDATE patients
                SET last_updated_at = ?,
                    latest_risk_level = COALESCE(?, latest_risk_level),
                    latest_ml_score = COALESCE(?, latest_ml_score),
                    metadata = COALESCE(?, metadata)
                WHERE patient_id = ?
            """, (now,
                  str(latest_risk_level) if latest_risk_level else None,
                  float(latest_ml_score) if latest_ml_score is not None else None,
                  json.dumps(metadata) if metadata else None,
                  patient_id))
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO patients (
                    patient_id, first_seen_at, last_updated_at,
                    latest_risk_level, latest_ml_score, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (patient_id, now, now,
                  str(latest_risk_level) if latest_risk_level else None,
                  float(latest_ml_score) if latest_ml_score is not None else None,
                  json.dumps(metadata) if metadata else None))

        conn.commit()
        conn.close()

        logger.debug(f"{'Updated' if exists else 'Created'} patient {patient_id}")

    def get_patient(self, patient_id: int) -> Optional[Dict]:
        """
        Get patient record.

        Args:
            patient_id: Patient identifier

        Returns:
            Patient data dictionary or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM patients WHERE patient_id = ?",
            (patient_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            patient = dict(row)
            if patient.get('metadata'):
                patient['metadata'] = json.loads(patient['metadata'])
            return patient
        return None

    def get_all_patients(self) -> pd.DataFrame:
        """
        Get all patients.

        Returns:
            DataFrame with all patients
        """
        conn = self._get_connection()
        df = pd.read_sql_query("SELECT * FROM patients ORDER BY patient_id", conn)
        conn.close()

        if len(df) > 0 and 'metadata' in df.columns:
            df['metadata'] = df['metadata'].apply(
                lambda x: json.loads(x) if x else None
            )

        return df

    # =========================================================================
    # Prediction Operations
    # =========================================================================

    def save_prediction(
        self,
        patient_id: int,
        ml_score: float,
        predicted_label: str,
        probabilities: Dict[str, float],
        feature_values: Dict[str, float] = None
    ) -> int:
        """
        Save ML prediction result.

        Args:
            patient_id: Patient identifier
            ml_score: ML risk score (0-1)
            predicted_label: Predicted label
            probabilities: Class probabilities
            feature_values: Feature values used for prediction

        Returns:
            Prediction record ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO predictions (
                patient_id, ml_score, predicted_label, probabilities,
                feature_values, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            int(patient_id),  # Ensure Python int
            float(ml_score),  # Ensure Python float
            str(predicted_label),
            json.dumps(probabilities),
            json.dumps(feature_values) if feature_values else None,
            now
        ))

        prediction_id = cursor.lastrowid

        # Update patient record
        cursor.execute("""
            UPDATE patients
            SET last_updated_at = ?, latest_ml_score = ?
            WHERE patient_id = ?
        """, (now, float(ml_score), int(patient_id)))

        conn.commit()
        conn.close()

        logger.debug(f"Saved prediction for patient {patient_id}: {ml_score:.3f}")

        return prediction_id

    def get_latest_prediction(self, patient_id: int) -> Optional[Dict]:
        """
        Get latest prediction for patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Prediction dictionary or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM predictions
            WHERE patient_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (patient_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            prediction = dict(row)
            prediction['probabilities'] = json.loads(prediction['probabilities'])
            if prediction.get('feature_values'):
                prediction['feature_values'] = json.loads(prediction['feature_values'])
            return prediction
        return None

    def get_prediction_history(
        self,
        patient_id: int,
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Get prediction history for patient.

        Args:
            patient_id: Patient identifier
            limit: Maximum number of records

        Returns:
            DataFrame with prediction history
        """
        conn = self._get_connection()

        df = pd.read_sql_query("""
            SELECT * FROM predictions
            WHERE patient_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, conn, params=(patient_id, limit))

        conn.close()

        if len(df) > 0:
            df['probabilities'] = df['probabilities'].apply(json.loads)
            df['feature_values'] = df['feature_values'].apply(
                lambda x: json.loads(x) if x else None
            )

        return df

    # =========================================================================
    # Stratification Operations
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
        """
        Save risk stratification result.

        Args:
            patient_id: Patient identifier
            risk_level: Final risk level
            ml_score: ML score
            threshold_based_level: Level before overrides
            override_applied: Whether override was applied
            override_reason: Override reason if applied
            recommendations: List of recommendations
            risk_metadata: Risk metadata (color, icon, etc.)

        Returns:
            Stratification record ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO stratifications (
                patient_id, risk_level, ml_score, threshold_based_level,
                override_applied, override_reason, recommendations,
                risk_metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(patient_id),  # Ensure Python int
            str(risk_level),
            float(ml_score),  # Ensure Python float
            str(threshold_based_level),
            int(override_applied),
            str(override_reason) if override_reason else None,
            json.dumps(recommendations) if recommendations else None,
            json.dumps(risk_metadata) if risk_metadata else None,
            now
        ))

        stratification_id = cursor.lastrowid

        # Update patient record
        cursor.execute("""
            UPDATE patients
            SET last_updated_at = ?, latest_risk_level = ?
            WHERE patient_id = ?
        """, (now, str(risk_level), int(patient_id)))

        conn.commit()
        conn.close()

        logger.debug(f"Saved stratification for patient {patient_id}: {risk_level}")

        return stratification_id

    def get_latest_stratification(self, patient_id: int) -> Optional[Dict]:
        """
        Get latest stratification for patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Stratification dictionary or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM stratifications
            WHERE patient_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (patient_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            stratification = dict(row)
            stratification['override_applied'] = bool(stratification['override_applied'])
            if stratification.get('recommendations'):
                stratification['recommendations'] = json.loads(stratification['recommendations'])
            if stratification.get('risk_metadata'):
                stratification['risk_metadata'] = json.loads(stratification['risk_metadata'])
            return stratification
        return None

    def get_stratification_history(
        self,
        patient_id: int,
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Get stratification history for patient.

        Args:
            patient_id: Patient identifier
            limit: Maximum number of records

        Returns:
            DataFrame with stratification history
        """
        conn = self._get_connection()

        df = pd.read_sql_query("""
            SELECT * FROM stratifications
            WHERE patient_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, conn, params=(patient_id, limit))

        conn.close()

        if len(df) > 0:
            df['override_applied'] = df['override_applied'].astype(bool)
            df['recommendations'] = df['recommendations'].apply(
                lambda x: json.loads(x) if x else None
            )
            df['risk_metadata'] = df['risk_metadata'].apply(
                lambda x: json.loads(x) if x else None
            )

        return df

    # =========================================================================
    # Observation Operations
    # =========================================================================

    def save_observation_metadata(
        self,
        patient_id: int,
        metric_name: str,
        value: float,
        unit: str,
        observation_date: str,
        fhir_resource_id: str = None
    ) -> int:
        """
        Save observation metadata (not full FHIR resource).

        Args:
            patient_id: Patient identifier
            metric_name: Metric name (e.g., 'steps', 'heart_rate_avg')
            value: Metric value
            unit: Unit of measure
            observation_date: Observation date
            fhir_resource_id: FHIR resource ID if posted to server

        Returns:
            Observation record ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO observations (
                patient_id, fhir_resource_id, metric_name, value, unit,
                observation_date, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            int(patient_id),  # Ensure Python int, not numpy int64
            fhir_resource_id,
            str(metric_name),
            float(value),
            str(unit) if unit else '',
            str(observation_date),
            now
        ))

        observation_id = cursor.lastrowid

        # Update patient observation count
        cursor.execute("""
            UPDATE patients
            SET total_observations = total_observations + 1,
                last_updated_at = ?
            WHERE patient_id = ?
        """, (now, patient_id))

        conn.commit()
        conn.close()

        return observation_id

    def get_patient_observations(
        self,
        patient_id: int,
        metric_name: str = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get observations for patient.

        Args:
            patient_id: Patient identifier
            metric_name: Optional metric filter
            limit: Maximum number of records

        Returns:
            DataFrame with observations
        """
        conn = self._get_connection()

        if metric_name:
            df = pd.read_sql_query("""
                SELECT * FROM observations
                WHERE patient_id = ? AND metric_name = ?
                ORDER BY observation_date DESC
                LIMIT ?
            """, conn, params=(patient_id, metric_name, limit))
        else:
            df = pd.read_sql_query("""
                SELECT * FROM observations
                WHERE patient_id = ?
                ORDER BY observation_date DESC
                LIMIT ?
            """, conn, params=(patient_id, limit))

        conn.close()
        return df

    # =========================================================================
    # Processing Log
    # =========================================================================

    def log_operation(
        self,
        operation: str,
        status: str,
        message: str = None,
        patient_id: int = None
    ):
        """
        Log a processing operation.

        Args:
            operation: Operation name
            status: Status (success, error, warning)
            message: Optional message
            patient_id: Optional patient ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO processing_log (
                patient_id, operation, status, message, created_at
            ) VALUES (?, ?, ?, ?, ?)
        """, (patient_id, operation, status, message, now))

        conn.commit()
        conn.close()

    def get_processing_log(self, limit: int = 100) -> pd.DataFrame:
        """
        Get processing log.

        Args:
            limit: Maximum number of records

        Returns:
            DataFrame with log entries
        """
        conn = self._get_connection()

        df = pd.read_sql_query("""
            SELECT * FROM processing_log
            ORDER BY created_at DESC
            LIMIT ?
        """, conn, params=(limit,))

        conn.close()
        return df

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}

        # Patient count
        cursor.execute("SELECT COUNT(*) FROM patients")
        stats['total_patients'] = cursor.fetchone()[0]

        # Risk level distribution
        cursor.execute("""
            SELECT latest_risk_level, COUNT(*)
            FROM patients
            WHERE latest_risk_level IS NOT NULL
            GROUP BY latest_risk_level
        """)
        stats['risk_distribution'] = dict(cursor.fetchall())

        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        stats['total_predictions'] = cursor.fetchone()[0]

        # Total stratifications
        cursor.execute("SELECT COUNT(*) FROM stratifications")
        stats['total_stratifications'] = cursor.fetchone()[0]

        # Total observations
        cursor.execute("SELECT COUNT(*) FROM observations")
        stats['total_observations'] = cursor.fetchone()[0]

        conn.close()

        return stats

    def clear_cache(self, confirm: bool = False):
        """
        Clear all cache data.

        Args:
            confirm: Must be True to actually clear
        """
        if not confirm:
            raise CacheError("Must confirm cache clear with confirm=True")

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM observations")
        cursor.execute("DELETE FROM stratifications")
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM patients")
        cursor.execute("DELETE FROM processing_log")

        conn.commit()
        conn.close()

        logger.warning("Cache cleared")


def init_database(db_path: str = None) -> SQLiteCache:
    """
    Initialize SQLite cache database.

    Args:
        db_path: Optional database path

    Returns:
        SQLiteCache instance
    """
    cache = SQLiteCache(db_path)
    logger.info("Database initialized")
    return cache


# Example usage
if __name__ == "__main__":
    print("=== SQLite Cache Demo ===\n")

    # Initialize cache
    cache = SQLiteCache()

    # Get stats
    stats = cache.get_stats()
    print("Cache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save sample patient
    print("\n1. Creating patient record...")
    cache.upsert_patient(
        patient_id=123,
        latest_risk_level="Yellow",
        latest_ml_score=0.55,
        metadata={"age": 45, "gender": "M"}
    )

    patient = cache.get_patient(123)
    print(f"Patient 123: {patient}")

    # Save prediction
    print("\n2. Saving prediction...")
    cache.save_prediction(
        patient_id=123,
        ml_score=0.55,
        predicted_label="Medium Risk",
        probabilities={"Low": 0.25, "Medium": 0.50, "High": 0.25},
        feature_values={"resting_hr_estimate": 82, "sleep_hours_avg": 6.5}
    )

    latest_pred = cache.get_latest_prediction(123)
    print(f"Latest prediction: ML score = {latest_pred['ml_score']:.3f}")

    # Save stratification
    print("\n3. Saving stratification...")
    cache.save_stratification(
        patient_id=123,
        risk_level="Yellow",
        ml_score=0.55,
        threshold_based_level="Yellow",
        override_applied=False,
        recommendations=[
            "Increase physical activity",
            "Improve sleep hygiene"
        ],
        risk_metadata={"color": "#ffc107", "icon": "⚠"}
    )

    latest_strat = cache.get_latest_stratification(123)
    print(f"Latest stratification: {latest_strat['risk_level']}")
    print(f"Recommendations: {latest_strat['recommendations']}")

    # Save observations
    print("\n4. Saving observations...")
    cache.save_observation_metadata(
        patient_id=123,
        metric_name="heart_rate_avg",
        value=82.0,
        unit="bpm",
        observation_date="2023-01-01"
    )

    cache.save_observation_metadata(
        patient_id=123,
        metric_name="steps",
        value=8500.0,
        unit="steps",
        observation_date="2023-01-01"
    )

    observations = cache.get_patient_observations(123)
    print(f"Patient 123 has {len(observations)} observations")

    # Get all patients
    print("\n5. Listing all patients...")
    patients_df = cache.get_all_patients()
    print(patients_df[['patient_id', 'latest_risk_level', 'latest_ml_score', 'total_observations']])

    # Log operation
    print("\n6. Logging operation...")
    cache.log_operation(
        operation="demo_test",
        status="success",
        message="Cache demo completed successfully"
    )

    # Final stats
    print("\n7. Final cache statistics:")
    final_stats = cache.get_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
