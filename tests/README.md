# CardioGuard Test Suite

Comprehensive unit and integration tests for CardioGuard.

## Test Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── unit/                    # Unit tests (individual modules)
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_label_generator.py
│   ├── test_risk_stratification.py
│   └── test_fhir_converter.py
└── integration/             # Integration tests (multi-module workflows)
    └── test_end_to_end.py
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Category

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_preprocessing.py

# Specific test class
pytest tests/unit/test_preprocessing.py::TestSchemaValidation

# Specific test function
pytest tests/unit/test_preprocessing.py::TestSchemaValidation::test_valid_schema
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Coverage Report

```bash
# Install pytest-cov first: pip install pytest-cov

pytest --cov=src --cov-report=html
```

Then open `htmlcov/index.html` to view coverage report.

### Run Tests Matching Pattern

```bash
pytest -k "test_clean"  # Run all tests with "clean" in name
pytest -k "preprocessing"  # Run all preprocessing tests
```

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- `sample_raw_data`: 30 days of synthetic fitness tracker data for 1 user
- `sample_features`: Pre-engineered cardiovascular features
- `sample_labels`: Synthetic risk labels
- `temp_db_path`: Temporary SQLite database path
- `mock_fhir_client`: Mocked FHIR client for testing

## Test Coverage

### Unit Tests

**test_preprocessing.py** (Schema validation, missing values, outliers, data cleaning)
- ✓ Schema validation with required columns
- ✓ Missing value handling strategies
- ✓ Outlier detection (IQR and Z-score methods)
- ✓ Complete data cleaning pipeline

**test_feature_engineering.py** (Cardiovascular feature calculations)
- ✓ Resting heart rate estimation
- ✓ Activity score calculation
- ✓ Rolling averages (7-day, 30-day)
- ✓ Sedentary ratio
- ✓ Workout consistency
- ✓ Heart rate variability
- ✓ Mood stress ratio

**test_label_generator.py** (Synthetic label generation)
- ✓ High risk label conditions
- ✓ Medium risk label conditions
- ✓ Low risk label conditions
- ✓ Label distribution validation

**test_risk_stratification.py** (ML score to risk level mapping)
- ✓ Threshold-based stratification (Green/Yellow/Red)
- ✓ Rule-based overrides
- ✓ Recommendation generation
- ✓ Condition evaluation

**test_fhir_converter.py** (Data to FHIR Observation conversion)
- ✓ LOINC code mapping
- ✓ Observation resource creation
- ✓ Batch conversion
- ✓ Missing value handling

### Integration Tests

**test_end_to_end.py** (Complete workflows)
- ✓ Full data pipeline (raw → cleaned → features)
- ✓ Full ML pipeline (features → training → prediction)
- ✓ Risk workflow (prediction → stratification)
- ✓ FHIR workflow (data → FHIR observations)
- ✓ Storage workflow (cache operations)
- ✓ Complete patient processing (end-to-end)

## Expected Test Results

With provided fixtures:
- **Total tests**: ~60
- **Expected pass rate**: 100%
- **Expected coverage**: ~70-80%

## Continuous Integration

To run tests in CI/CD:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.13'
      - run: pip install -r requirements.txt
      - run: pytest --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Troubleshooting

**Import errors:**
- Make sure you're running pytest from project root
- Ensure all dependencies are installed: `pip install -r requirements.txt`

**FHIR-related test failures:**
- FHIR integration tests will skip if FHIR server is not available
- Use `--no-fhir` flag or skip integration tests in CI

**Database conflicts:**
- Tests use temporary databases (pytest tmp_path fixture)
- No cleanup needed - pytest handles automatically

## Writing New Tests

### Unit Test Template

```python
"""
Unit tests for new_module.
"""

import pytest
from src.module import function_to_test


class TestNewFeature:
    """Tests for new feature."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        result = function_to_test(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """Test edge case."""
        # ... test code
```

### Integration Test Template

```python
"""
Integration tests for new workflow.
"""

import pytest


class TestNewWorkflow:
    """Integration tests for new workflow."""

    def test_complete_workflow(self, fixture1, fixture2):
        """Test complete workflow."""
        # Step 1
        result1 = step1(fixture1)

        # Step 2
        result2 = step2(result1)

        # Assertions
        assert result2.is_valid()
```

## Best Practices

1. **One assertion per test** (when possible) - Makes failures easier to diagnose
2. **Use descriptive test names** - `test_handle_missing_sleep` vs `test_function1`
3. **Arrange-Act-Assert pattern** - Clear test structure
4. **Use fixtures** - Reuse test data across tests
5. **Test edge cases** - Empty inputs, None values, boundary conditions
6. **Mock external dependencies** - Don't rely on FHIR server for unit tests

## Getting Help

- pytest documentation: https://docs.pytest.org/
- Issue tracker: https://github.com/anthropics/cardioguard/issues
