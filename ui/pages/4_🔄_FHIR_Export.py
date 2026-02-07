"""
FHIR Export Page

View and download FHIR-compliant resources.
"""

import streamlit as st
import json
import pandas as pd

from src.auth.simple_auth import require_authentication
from src.storage.sqlite_cache import SQLiteCache
from src.fhir.converter import create_observation
from src.fhir.risk_resources import create_risk_assessment, create_risk_flag
from ui.components.disclaimer import show_disclaimer, show_footer
from src.utils.constants import LOINC_CODES, RISK_LEVEL_RED, RISK_LEVEL_YELLOW


# Page configuration
st.set_page_config(
    page_title="FHIR Export - CardioGuard",
    page_icon="🔄",
    layout="wide"
)

# Require authentication
require_authentication()

# Header
st.title("🔄 FHIR Export")
show_disclaimer()

st.markdown("---")

# Introduction
st.markdown("""
### HL7 FHIR R4 Interoperability

This page demonstrates **FHIR (Fast Healthcare Interoperability Resources)** standard compliance.

**FHIR Resources Generated:**
- **Observation** - Wearable device metrics with LOINC codes
- **RiskAssessment** - ML-based risk predictions with SHAP explanations
- **Flag** - Clinical alerts for medium/high risk patients

All resources include educational disclaimers and metadata tags.
""")

st.markdown("---")

# Load data
try:
    cache = SQLiteCache()
    patients_df = cache.get_all_patients()

    if len(patients_df) == 0:
        st.warning("No patients found. Please run the training script first.")
        st.stop()

    # Patient selection
    selected_patient_id = st.selectbox(
        "Select Patient",
        options=patients_df['patient_id'].tolist(),
        index=0,
        format_func=lambda x: f"Patient {x}"
    )

    # Load patient data
    patient = cache.get_patient(selected_patient_id)
    observations = cache.get_patient_observations(selected_patient_id, limit=100)
    latest_strat = cache.get_latest_stratification(selected_patient_id)

    st.markdown(f"**Patient {selected_patient_id}** - {len(observations)} observations available")

    st.markdown("---")

    # Resource type selection
    st.markdown("### 📋 Select FHIR Resource Type")

    resource_type = st.radio(
        "Choose resource to view",
        options=["Observation", "RiskAssessment", "Flag"],
        horizontal=True
    )

    st.markdown("---")

    # OBSERVATION
    if resource_type == "Observation":
        st.markdown("### 🔬 FHIR Observation Resources")

        st.info("""
        **Observation Resources** represent wearable device measurements.
        Each observation is mapped to a standard LOINC code for interoperability.
        """)

        # LOINC code reference
        with st.expander("📚 LOINC Code Reference"):
            loinc_df = pd.DataFrame([
                {'Metric': k.replace('_', ' ').title(), 'LOINC Code': v}
                for k, v in LOINC_CODES.items()
            ])
            st.dataframe(loinc_df, use_container_width=True, hide_index=True)

        if len(observations) > 0:
            # Metric selection
            available_metrics = observations['metric_name'].unique().tolist()

            selected_metric = st.selectbox(
                "Select Metric",
                options=available_metrics,
                format_func=lambda x: x.replace('_', ' ').title()
            )

            # Get a sample observation
            metric_obs = observations[observations['metric_name'] == selected_metric].iloc[0]

            # Create FHIR Observation
            fhir_obs = create_observation(
                user_id=selected_patient_id,
                date=metric_obs['observation_date'],
                metric_name=metric_obs['metric_name'],
                value=metric_obs['value']
            )

            # Display JSON
            st.markdown("#### FHIR JSON")

            obs_json = json.loads(fhir_obs.json())

            st.json(obs_json, expanded=True)

            # Download button
            st.download_button(
                label="📥 Download Observation JSON",
                data=fhir_obs.json(indent=2),
                file_name=f"observation_{selected_patient_id}_{selected_metric}.json",
                mime="application/json"
            )

            # Key fields explanation
            with st.expander("🔍 Key FHIR Fields Explained"):
                st.markdown(f"""
                - **resourceType**: `Observation` - FHIR resource type
                - **status**: `{obs_json['status']}` - Observation status
                - **code.coding[0].system**: `http://loinc.org` - Standard coding system
                - **code.coding[0].code**: `{obs_json['code']['coding'][0]['code']}` - LOINC code
                - **subject.reference**: `Patient/{selected_patient_id}` - Patient reference
                - **effectiveDateTime**: `{obs_json['effectiveDateTime']}` - When measured
                - **valueQuantity.value**: `{obs_json['valueQuantity']['value']}` - Measured value
                - **valueQuantity.unit**: `{obs_json['valueQuantity']['unit']}` - Unit of measure
                """)

        else:
            st.info("No observations available for this patient")

    # RISK ASSESSMENT
    elif resource_type == "RiskAssessment":
        st.markdown("### 🎯 FHIR RiskAssessment Resource")

        st.info("""
        **RiskAssessment Resources** represent ML-based risk predictions.
        Includes probability scores, risk levels, and SHAP explanations in notes.
        """)

        if latest_strat:
            # Create FHIR RiskAssessment (with mock SHAP features)
            mock_features = [
                {
                    'feature_name': 'resting_hr_estimate',
                    'feature_display_name': 'Resting Heart Rate',
                    'value': 85.0,
                    'unit': 'bpm',
                    'impact': 'INCREASES',
                    'shap_value': 0.25
                },
                {
                    'feature_name': 'sleep_hours_avg',
                    'feature_display_name': 'Average Sleep Duration',
                    'value': 6.2,
                    'unit': 'hours',
                    'impact': 'INCREASES',
                    'shap_value': 0.18
                }
            ]

            fhir_risk = create_risk_assessment(
                user_id=selected_patient_id,
                ml_score=latest_strat['ml_score'],
                risk_level=latest_strat['risk_level'],
                top_features=mock_features
            )

            # Display JSON
            st.markdown("#### FHIR JSON")

            risk_json = json.loads(fhir_risk.json())

            st.json(risk_json, expanded=True)

            # Download button
            st.download_button(
                label="📥 Download RiskAssessment JSON",
                data=fhir_risk.json(indent=2),
                file_name=f"riskassessment_{selected_patient_id}.json",
                mime="application/json"
            )

            # Key fields explanation
            with st.expander("🔍 Key FHIR Fields Explained"):
                st.markdown(f"""
                - **resourceType**: `RiskAssessment` - FHIR resource type
                - **status**: `{risk_json['status']}` - Assessment status
                - **subject.reference**: `Patient/{selected_patient_id}` - Patient reference
                - **occurrenceDateTime**: When assessment was performed
                - **prediction[0].outcome.text**: `{risk_json['prediction'][0]['outcome']['text']}` - What is being assessed
                - **prediction[0].probabilityDecimal**: `{risk_json['prediction'][0]['probabilityDecimal']:.3f}` - ML risk score
                - **prediction[0].qualitativeRisk.text**: Risk level description
                - **note**: SHAP feature explanations and disclaimers
                - **meta.tag**: Educational disclaimer tag (HTEST - test health data)
                """)

        else:
            st.info("No risk stratification available for this patient")

    # FLAG
    elif resource_type == "Flag":
        st.markdown("### 🚩 FHIR Flag Resource")

        st.info("""
        **Flag Resources** represent clinical alerts for medium/high risk patients.
        Green (low risk) patients do not generate flags.
        """)

        if latest_strat:
            risk_level = latest_strat['risk_level']

            if risk_level in [RISK_LEVEL_YELLOW, RISK_LEVEL_RED]:
                # Create FHIR Flag
                fhir_flag = create_risk_flag(
                    user_id=selected_patient_id,
                    risk_level=risk_level,
                    reason=f"Cardiovascular wellness risk: {risk_level}"
                )

                # Display JSON
                st.markdown("#### FHIR JSON")

                flag_json = json.loads(fhir_flag.json())

                st.json(flag_json, expanded=True)

                # Download button
                st.download_button(
                    label="📥 Download Flag JSON",
                    data=fhir_flag.json(indent=2),
                    file_name=f"flag_{selected_patient_id}.json",
                    mime="application/json"
                )

                # Key fields explanation
                with st.expander("🔍 Key FHIR Fields Explained"):
                    st.markdown(f"""
                    - **resourceType**: `Flag` - FHIR resource type
                    - **status**: `{flag_json['status']}` - Flag status (active)
                    - **category[0].coding[0].code**: `{flag_json['category'][0]['coding'][0]['code']}` - Clinical category
                    - **code.text**: `{flag_json['code']['text']}` - Flag description
                    - **subject.reference**: `Patient/{selected_patient_id}` - Patient reference
                    - **period.start**: When flag became active
                    - **meta.tag**: Educational disclaimer tag
                    """)

            else:
                st.success(f"""
                **No Flag Generated**

                Patient {selected_patient_id} has **{risk_level}** risk level.

                Flags are only created for Yellow (medium) or Red (high) risk patients.
                Green (low risk) patients do not require clinical flags.
                """)

        else:
            st.info("No risk stratification available for this patient")

    st.markdown("---")

    # Bulk export
    st.markdown("### 📦 Bulk Export")

    st.info("""
    Export all FHIR resources for this patient as a FHIR Bundle.

    Note: In a production system, this would generate a proper FHIR Bundle resource.
    For this demonstration, resources are exported individually.
    """)

    if st.button("Generate Bundle Export", type="primary"):
        bundle_data = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": []
        }

        # Add observations
        for idx, obs_row in observations.head(5).iterrows():
            fhir_obs = create_observation(
                user_id=selected_patient_id,
                date=obs_row['observation_date'],
                metric_name=obs_row['metric_name'],
                value=obs_row['value']
            )

            bundle_data["entry"].append({
                "resource": json.loads(fhir_obs.json())
            })

        # Add RiskAssessment
        if latest_strat:
            fhir_risk = create_risk_assessment(
                user_id=selected_patient_id,
                ml_score=latest_strat['ml_score'],
                risk_level=latest_strat['risk_level']
            )

            bundle_data["entry"].append({
                "resource": json.loads(fhir_risk.json())
            })

        st.json(bundle_data, expanded=False)

        st.download_button(
            label="📥 Download Bundle",
            data=json.dumps(bundle_data, indent=2),
            file_name=f"bundle_patient_{selected_patient_id}.json",
            mime="application/json"
        )

except Exception as e:
    st.error(f"Error loading FHIR resources: {e}")

    import traceback
    with st.expander("Error Details"):
        st.code(traceback.format_exc())

show_footer()
