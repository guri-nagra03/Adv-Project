"""
Constants for CardioGuard

Centralized constants used across the application.
"""

# FHIR Resource Types
FHIR_OBSERVATION = "Observation"
FHIR_RISK_ASSESSMENT = "RiskAssessment"
FHIR_FLAG = "Flag"
FHIR_PATIENT = "Patient"

# LOINC Codes for Wearable Metrics
LOINC_CODES = {
    "steps": "41950-7",  # Number of steps in 24 hour Measured
    "heart_rate_avg": "8867-4",  # Heart rate
    "sleep_hours": "93832-4",  # Sleep duration
    "active_minutes": "82290-8",  # Frequency of physical activity
    "calories_burned": "41979-6",  # Calories burned
    "distance_km": "41953-1"  # Distance traveled
}

# LOINC Display Names
LOINC_DISPLAY_NAMES = {
    "steps": "Number of steps in 24 hour",
    "heart_rate_avg": "Heart rate",
    "sleep_hours": "Sleep duration",
    "active_minutes": "Frequency of physical activity",
    "calories_burned": "Calories burned",
    "distance_km": "Distance traveled"
}

# Units of Measure
UNITS = {
    "steps": "steps",
    "heart_rate_avg": "bpm",
    "sleep_hours": "h",
    "active_minutes": "min",
    "calories_burned": "kcal",
    "distance_km": "km"
}

# Risk Levels
RISK_LEVEL_GREEN = "Green"
RISK_LEVEL_YELLOW = "Yellow"
RISK_LEVEL_RED = "Red"

RISK_LEVELS = [RISK_LEVEL_GREEN, RISK_LEVEL_YELLOW, RISK_LEVEL_RED]

# Risk Level Numeric Mapping (for ML labels)
RISK_LABEL_MAPPING = {
    0: "Low",
    1: "Medium",
    2: "High"
}

RISK_LABEL_TO_COLOR = {
    "Low": RISK_LEVEL_GREEN,
    "Medium": RISK_LEVEL_YELLOW,
    "High": RISK_LEVEL_RED
}

# Dataset Column Names
REQUIRED_COLUMNS = [
    "user_id",
    "date",
    "steps",
    "calories_burned",
    "distance_km",
    "active_minutes",
    "sleep_hours",
    "heart_rate_avg",
    "workout_type",
    "mood"
]

# Feature Names (derived from feature engineering)
FEATURE_NAMES = [
    "resting_hr_estimate",
    "activity_score",
    "sleep_hours_avg",
    "sedentary_ratio",
    "workout_consistency",
    "hr_variability_proxy",
    "mood_stress_ratio"
]

# Healthy Ranges (for feature interpretation)
HEALTHY_RANGES = {
    "resting_hr_estimate": (60, 80),
    "activity_score": (50, 100),
    "sleep_hours_avg": (7, 9),
    "sedentary_ratio": (0, 0.3),
    "workout_consistency": (0.5, 1.0),
    "hr_variability_proxy": (5, 15),
    "mood_stress_ratio": (0, 0.3)
}

# Mood Categories
MOOD_STRESSED = "Stressed"
MOOD_HAPPY = "Happy"
MOOD_NEUTRAL = "Neutral"
MOOD_TIRED = "Tired"

# Workout Types
WORKOUT_TYPES = [
    "Walking",
    "Running",
    "Cycling",
    "Swimming",
    "Gym Workout",
    "Yoga",
    "None"
]

# SQLite Table Names
TABLE_PATIENTS = "patients"
TABLE_PROCESSED_DATA = "processed_data"
TABLE_RISK_HISTORY = "risk_history"

# FHIR Flag Categories
FLAG_CATEGORY_CLINICAL = "clinical"

# FHIR Status Values
STATUS_FINAL = "final"
STATUS_ACTIVE = "active"

# Educational Disclaimer
DISCLAIMER_BANNER = (
    "⚠️ **EDUCATIONAL DEMONSTRATION ONLY**\n\n"
    "This application does NOT provide medical advice, diagnosis, or treatment.\n"
    "For medical concerns, consult a qualified healthcare provider."
)

# Color Scheme
COLORS = {
    "green": "#28a745",
    "yellow": "#ffc107",
    "red": "#dc3545",
    "blue": "#007bff",
    "gray": "#6c757d"
}

# Icons
ICONS = {
    "low_risk": "✓",
    "medium_risk": "⚠",
    "high_risk": "⚠⚠",
    "heart": "❤️",
    "warning": "⚠️",
    "info": "ℹ️"
}
