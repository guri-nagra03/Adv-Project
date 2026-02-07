# CardioGuard

**Preventive Cardiovascular Wellness Monitoring System**

> ⚠️ **EDUCATIONAL DEMONSTRATION ONLY**
> This application does NOT provide medical advice, diagnosis, or treatment.
> For medical concerns, consult a qualified healthcare provider.

## Overview

CardioGuard is a student-level health informatics application demonstrating:
- **HL7 FHIR Interoperability**: Convert wearable data to standards-compliant FHIR resources
- **Explainable ML**: Logistic regression with SHAP explanations for cardiovascular wellness risk scoring
- **Rule-Based Risk Stratification**: Transparent Green/Yellow/Red categorization
- **Clinician Dashboard**: Streamlit web interface for patient monitoring

**Key Principle**: This is an educational demonstration of health informatics concepts, NOT a medical diagnostic tool.

## Features

- 📊 **Data Ingestion**: CSV upload + simulated streaming from wearable devices
- 🧠 **Machine Learning**: Simple, interpretable model for risk scoring with feature explanations
- 🏥 **FHIR Integration**: Observations, RiskAssessments, and Flags sent to HAPI FHIR server
- 🎨 **Interactive Dashboard**: Patient list, risk visualizations, trends, and FHIR export
- 🔒 **Basic Authentication**: Demo clinician login
- 🐳 **Docker Deployment**: Single command to start HAPI FHIR + CardioGuard

## Architecture

```
Wearable Data (CSV) → Preprocessing → Feature Engineering → ML Model
                                                              ↓
                                                    Risk Stratification
                                                     ↓              ↓
                                            FHIR Converter    Streamlit UI
                                                     ↓
                                              HAPI FHIR Server
```

## Tech Stack

- **Python 3.13**
- **ML**: scikit-learn (Logistic Regression), SHAP (Explainability)
- **FHIR**: fhir.resources (Pydantic validation), HAPI FHIR server (Docker)
- **UI**: Streamlit, Plotly
- **Storage**: SQLite (caching), FHIR server (primary)
- **Deployment**: Docker Compose

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.13+ (for local development)

### Setup

1. **Clone and navigate to the project**:
   ```bash
   cd /Users/gurinagra/Codes/Python/cardioguard
   ```

2. **Start services with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

   This will start:
   - HAPI FHIR Server at http://localhost:8080
   - CardioGuard App at http://localhost:8501

3. **Access the application**:
   - **CardioGuard Dashboard**: http://localhost:8501
   - **HAPI FHIR Server UI**: http://localhost:8080

4. **Login credentials** (demo):
   - Username: `clinician1`
   - Password: `demo123`

### Local Development (without Docker)

1. **Create virtual environment**:
   ```bash
   python3.13 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start HAPI FHIR server** (Docker):
   ```bash
   docker run -p 8080:8080 hapiproject/hapi:latest
   ```

4. **Train the ML model**:
   ```bash
   python scripts/train_model.py
   ```

5. **Run the Streamlit app**:
   ```bash
   streamlit run ui/app.py
   ```

## Data Pipeline

### 1. Feature Engineering

Cardiovascular wellness features derived from wearable data:

| Feature | Description |
|---------|-------------|
| `resting_hr_estimate` | Heart rate during low-activity periods (steps < 1000) |
| `activity_score` | Weighted combination: steps (40%) + active_minutes (40%) + distance (20%) |
| `sleep_hours_avg` | 7-day rolling average |
| `sedentary_ratio` | Proportion of days with < 5000 steps (last 30 days) |
| `workout_consistency` | Proportion of days with workouts (last 30 days) |
| `hr_variability_proxy` | Standard deviation of heart rate (7-day window) |
| `mood_stress_ratio` | Proportion of "Stressed" mood (last 14 days) |

### 2. Synthetic Label Generation

Since we lack ground truth cardiovascular outcomes, we create educational labels using domain knowledge:

- **HIGH RISK**: Resting HR > 90 bpm OR sleep < 6 hours OR activity score < 20th percentile OR sedentary ratio > 70%
- **MEDIUM RISK**: Resting HR 75-90 bpm OR sleep 6-7 hours OR activity score 20th-50th percentile
- **LOW RISK**: Otherwise

### 3. ML Model

- **Algorithm**: Multinomial Logistic Regression
- **Why**: Interpretable, fast, probabilistic outputs
- **Explainability**: SHAP values identify top contributing features

### 4. Risk Stratification

| Category | ML Score | Color | Recommendations |
|----------|----------|-------|----------------|
| Green (Low) | < 0.33 | 🟢 | Maintain activity, monitor monthly |
| Yellow (Medium) | 0.33 - 0.66 | 🟡 | Increase activity to 150 min/week, improve sleep |
| Red (High) | > 0.66 | 🔴 | Schedule wellness consultation, track HR daily |

**Rule Overrides** (force Red):
- Steps avg (30d) < 2000
- Sleep avg (7d) < 5 hours
- Resting HR > 100 bpm

## FHIR Interoperability

CardioGuard converts wearable data to HL7 FHIR R4 resources:

### Observation Mappings

| Wearable Metric | LOINC Code | FHIR Resource |
|-----------------|------------|---------------|
| Steps | 41950-7 | Observation |
| Heart Rate | 8867-4 | Observation |
| Sleep Duration | 93832-4 | Observation |
| Active Minutes | 82290-8 | Observation |
| Calories Burned | 41979-6 | Observation |
| Distance | 41953-1 | Observation |

### Risk Resources

- **RiskAssessment**: ML score, risk level, basis (Observation references), SHAP explanations
- **Flag**: Created for Yellow/Red patients only (status: "active", category: "clinical")

### FHIR Server Interaction

- POST observations in batches of 100
- Retry logic: 3 attempts with exponential backoff
- Graceful degradation to local cache if FHIR unavailable

## UI Dashboard

### Pages

1. **Patient List**: Table view with risk levels (color-coded), filterable
2. **Risk Dashboard**: ML score gauge, top 3 contributing features, recommendations
3. **Trends**: Time-series charts (heart rate, activity, sleep), mood correlation
4. **FHIR Export**: JSON viewer, download FHIR Bundle

### Authentication

Simple demo authentication (hardcoded users):
- `clinician1` / `demo123`
- `admin` / `admin456`

## Testing

```bash
# Run all tests
pytest

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires Docker services running)
pytest tests/integration/ -v

# Coverage report
pytest --cov=src --cov-report=html
```

## Project Structure

```
cardioguard/
├── config/           # Configuration files
├── data/             # Raw and processed data
├── src/              # Source code
│   ├── data/         # Ingestion, preprocessing, feature engineering
│   ├── ml/           # Model training, prediction, explainability
│   ├── risk/         # Risk stratification, rule-based logic
│   ├── fhir/         # FHIR conversion, client, validators
│   ├── storage/      # SQLite cache, FHIR repository
│   ├── auth/         # Simple authentication
│   └── utils/        # Logging, constants
├── ui/               # Streamlit dashboard
│   ├── pages/        # Patient list, risk dashboard, trends, FHIR export
│   └── components/   # Reusable UI components (disclaimer, charts)
├── models/           # Trained ML models
├── tests/            # Unit and integration tests
├── scripts/          # Utility scripts (training, seeding)
└── docs/             # Documentation
```

## Ethical Guidelines

### Language Guidelines

✅ **Use**: "wellness risk indicator", "preventive monitoring", "activity pattern"
❌ **Avoid**: "diagnosis", "disease risk", "clinical decision support"

### Disclaimers

Every page displays:
> ⚠️ EDUCATIONAL DEMONSTRATION ONLY
> This application does NOT provide medical advice, diagnosis, or treatment.

### Data Safety

- Use only first 10,000 rows of dataset (reasonable demo size)
- No real patient data (synthetic user IDs)
- No emergency alerts (educational only)

## Configuration

### Environment Variables

Create `.env` file:

```env
FHIR_SERVER_URL=http://localhost:8080/fhir
DATASET_PATH=/app/data/raw/fitness_tracker_dataset.csv
MODEL_PATH=/app/models/risk_classifier.pkl
SQLITE_DB_PATH=/app/data/cache/cardioguard.db
DATA_LIMIT=10000
```

### Risk Thresholds

Edit `config/risk_thresholds.yaml` to adjust risk categorization rules.

## Documentation

- **Architecture**: [docs/architecture.md](docs/architecture.md)
- **FHIR Mappings**: [docs/fhir_mapping.md](docs/fhir_mapping.md)
- **User Guide**: [docs/user_guide.md](docs/user_guide.md)

## Contributing

This is an educational project. Contributions are welcome for:
- Improved FHIR resource mappings
- Additional cardiovascular risk features
- Enhanced visualizations
- Better explainability techniques

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- **FHIR**: HL7 FHIR R4 specification
- **HAPI FHIR**: Open-source FHIR server
- **Dataset**: Synthetic fitness tracker data for educational purposes

## Disclaimer

**THIS APPLICATION IS FOR EDUCATIONAL PURPOSES ONLY.**

CardioGuard is a student-level demonstration of health informatics concepts and is NOT:
- A medical device
- Cleared or approved by FDA or any regulatory body
- Intended for clinical diagnosis or treatment
- A substitute for professional medical advice

For medical concerns, consult a qualified healthcare provider.

## Support

For issues or questions about this educational project, please open an issue in the repository.

---

**Built with**: Python 🐍 | FHIR 🏥 | Streamlit 🎨 | Docker 🐳
