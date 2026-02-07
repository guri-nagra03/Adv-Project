# CardioGuard Scripts

Utility scripts for training, data processing, and setup.

## Training Script

**`train_model.py`** - Complete end-to-end training pipeline

### Usage

Basic training (with FHIR server):
```bash
python scripts/train_model.py
```

Cache-only mode (skip FHIR server):
```bash
python scripts/train_model.py --no-fhir
```

Custom data limit:
```bash
python scripts/train_model.py --limit 5000
```

Use existing model (skip training):
```bash
python scripts/train_model.py --skip-training
```

### What It Does

1. **Load Data** - Loads fitness tracker CSV (first 10,000 rows by default)
2. **Preprocess** - Cleans data, handles missing values, removes outliers
3. **Feature Engineering** - Creates 7 cardiovascular features
4. **Generate Labels** - Creates synthetic risk labels using rules
5. **Train Model** - Trains logistic regression classifier
6. **Generate Predictions** - Predicts risk for all patients
7. **Populate Database** - Saves to SQLite cache + FHIR server
8. **Create Resources** - Generates FHIR Observations, RiskAssessments, Flags

### Output

- **Model**: `models/risk_classifier.pkl`
- **Scaler**: `models/scaler.pkl`
- **Database**: `data/cache/cardioguard.db` (SQLite)
- **FHIR Resources**: Posted to HAPI FHIR server (if enabled)

### Expected Runtime

- **10,000 rows**: ~2-5 minutes (depends on FHIR server)
- **Cache-only mode**: ~1-2 minutes
- **With existing model**: ~1 minute

### Requirements

- Dataset at `data/raw/fitness_tracker_dataset.csv`
- FHIR server running (or use `--no-fhir`)
- Python dependencies installed (`pip install -r requirements.txt`)

### Troubleshooting

**"Dataset not found"**
- Check `data/raw/fitness_tracker_dataset.csv` exists
- Update path in `config/settings.py` if needed

**"FHIR server not accessible"**
- Start FHIR server: `docker-compose up -d fhir-server`
- Or use `--no-fhir` flag for cache-only mode

**"Model training failed"**
- Check data quality
- Ensure enough valid features (need at least 100 rows)
- Check logs for specific errors

## Next Steps

After running `train_model.py`:

1. **Start UI**:
   ```bash
   streamlit run ui/app.py
   ```

2. **Login**:
   - Username: `clinician1`
   - Password: `demo123`

3. **Explore**:
   - Patient List - Browse all patients
   - Risk Dashboard - Detailed patient analysis
   - Trends - Time-series visualizations
   - FHIR Export - View FHIR resources
