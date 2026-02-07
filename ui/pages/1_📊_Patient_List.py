"""
Patient List Page

View all patients with risk levels and basic information.
"""

import streamlit as st
import pandas as pd

from src.auth.simple_auth import require_authentication
from src.storage.sqlite_cache import SQLiteCache
from ui.components.disclaimer import show_disclaimer, show_footer
from ui.components.metrics import show_metric_card
from ui.components.charts import plot_risk_distribution


# Page configuration
st.set_page_config(
    page_title="Patient List - CardioGuard",
    page_icon="📊",
    layout="wide"
)

# Require authentication
require_authentication()

# Header
st.title("📊 Patient List")
show_disclaimer()

st.markdown("---")

# Load data
try:
    cache = SQLiteCache()
    patients_df = cache.get_all_patients()

    if len(patients_df) == 0:
        st.warning("""
        No patients found in the system.

        Please run the training script first to process data and generate predictions:

        ```bash
        python scripts/train_model.py
        ```
        """)
        st.stop()

    # Summary metrics
    st.markdown("### 📈 Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_metric_card(
            "Total Patients",
            str(len(patients_df)),
            icon="👥"
        )

    with col2:
        green_count = (patients_df['latest_risk_level'] == 'Green').sum()
        show_metric_card(
            "Green Risk",
            str(green_count),
            icon="✓"
        )

    with col3:
        yellow_count = (patients_df['latest_risk_level'] == 'Yellow').sum()
        show_metric_card(
            "Yellow Risk",
            str(yellow_count),
            icon="⚠"
        )

    with col4:
        red_count = (patients_df['latest_risk_level'] == 'Red').sum()
        show_metric_card(
            "Red Risk",
            str(red_count),
            icon="🔴"
        )

    st.markdown("---")

    # Risk distribution chart
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎯 Risk Level Distribution")

        # Filter controls
        col_a, col_b = st.columns(2)

        with col_a:
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                options=['Green', 'Yellow', 'Red'],
                default=['Green', 'Yellow', 'Red']
            )

        with col_b:
            sort_by = st.selectbox(
                "Sort by",
                options=['patient_id', 'latest_ml_score', 'latest_risk_level', 'last_updated_at'],
                index=1
            )

        # Filter data
        if risk_filter:
            filtered_df = patients_df[patients_df['latest_risk_level'].isin(risk_filter)]
        else:
            filtered_df = patients_df

        # Sort data
        filtered_df = filtered_df.sort_values(
            by=sort_by,
            ascending=(sort_by == 'patient_id')
        )

        # Patient table
        st.markdown(f"**Showing {len(filtered_df)} patients**")

        # Format table
        display_df = filtered_df[[
            'patient_id',
            'latest_risk_level',
            'latest_ml_score',
            'total_observations',
            'last_updated_at'
        ]].copy()

        display_df['latest_ml_score'] = display_df['latest_ml_score'].round(3)
        display_df['last_updated_at'] = pd.to_datetime(display_df['last_updated_at']).dt.strftime('%Y-%m-%d %H:%M')

        display_df.columns = ['Patient ID', 'Risk Level', 'ML Score', 'Observations', 'Last Updated']

        # Color code risk levels
        def color_risk_level(val):
            if val == 'Green':
                return 'background-color: #d4edda'
            elif val == 'Yellow':
                return 'background-color: #fff3cd'
            elif val == 'Red':
                return 'background-color: #f8d7da'
            return ''

        styled_df = display_df.style.applymap(
            color_risk_level,
            subset=['Risk Level']
        )

        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

    with col2:
        # Risk distribution pie chart
        if len(filtered_df) > 0:
            risk_counts = filtered_df['latest_risk_level'].value_counts().to_dict()
            plot_risk_distribution(risk_counts)
        else:
            st.info("No data to display")

    st.markdown("---")

    # Patient selection for detailed view
    st.markdown("### 🔍 View Patient Details")

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_patient_id = st.selectbox(
            "Select Patient ID",
            options=filtered_df['patient_id'].tolist(),
            format_func=lambda x: f"Patient {x}"
        )

    with col2:
        if st.button("View Risk Dashboard", type="primary", use_container_width=True):
            st.session_state['selected_patient_id'] = selected_patient_id
            st.switch_page("pages/2_🎯_Risk_Dashboard.py")

    # Show selected patient summary
    if selected_patient_id:
        patient = cache.get_patient(selected_patient_id)

        if patient:
            st.markdown(f"#### Patient {selected_patient_id} Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Risk Level", patient['latest_risk_level'])

            with col2:
                st.metric("ML Score", f"{patient['latest_ml_score']:.3f}")

            with col3:
                st.metric("Observations", patient['total_observations'])

            with col4:
                first_seen = pd.to_datetime(patient['first_seen_at']).strftime('%Y-%m-%d')
                st.metric("First Seen", first_seen)

            # Latest stratification
            strat = cache.get_latest_stratification(selected_patient_id)
            if strat:
                st.markdown("**Latest Recommendations:**")
                for rec in strat.get('recommendations', [])[:3]:
                    st.markdown(f"- {rec}")

except Exception as e:
    st.error(f"Error loading patient data: {e}")
    st.info("Make sure the database is initialized and data is loaded.")

    import traceback
    with st.expander("Error Details"):
        st.code(traceback.format_exc())

show_footer()
