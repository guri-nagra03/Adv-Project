"""
Risk Dashboard Page

Detailed patient risk analysis with ML explanations and recommendations.
"""

import streamlit as st
import pandas as pd

from src.auth.simple_auth import require_authentication
from src.storage.sqlite_cache import SQLiteCache
from ui.components.disclaimer import show_disclaimer, show_footer
from ui.components.metrics import (
    show_risk_badge,
    show_ml_score_gauge,
    show_recommendations_list,
    show_feature_explanation
)
from ui.components.charts import plot_feature_importance


# Page configuration
st.set_page_config(
    page_title="Risk Dashboard - CardioGuard",
    page_icon="🎯",
    layout="wide"
)

# Require authentication
require_authentication()

# Header
st.title("🎯 Risk Dashboard")
show_disclaimer()

st.markdown("---")

# Load data
try:
    cache = SQLiteCache()
    patients_df = cache.get_all_patients()

    if len(patients_df) == 0:
        st.warning("No patients found. Please run the training script first.")
        st.stop()

    # Patient selection
    col1, col2 = st.columns([3, 1])

    with col1:
        # Check if patient was selected from Patient List page
        if 'selected_patient_id' in st.session_state:
            default_patient = st.session_state['selected_patient_id']
        else:
            default_patient = patients_df.iloc[0]['patient_id']

        selected_patient_id = st.selectbox(
            "Select Patient",
            options=patients_df['patient_id'].tolist(),
            index=patients_df['patient_id'].tolist().index(default_patient) if default_patient in patients_df['patient_id'].tolist() else 0,
            format_func=lambda x: f"Patient {x}"
        )

        # Update session state
        st.session_state['selected_patient_id'] = selected_patient_id

    with col2:
        if st.button("← Back to Patient List", use_container_width=True):
            st.switch_page("pages/1_📊_Patient_List.py")

    st.markdown("---")

    # Load patient data
    patient = cache.get_patient(selected_patient_id)
    latest_pred = cache.get_latest_prediction(selected_patient_id)
    latest_strat = cache.get_latest_stratification(selected_patient_id)

    if not patient:
        st.error(f"Patient {selected_patient_id} not found")
        st.stop()

    # Patient info header
    st.markdown(f"### Patient {selected_patient_id}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Observations", patient['total_observations'])

    with col2:
        first_seen = pd.to_datetime(patient['first_seen_at']).strftime('%Y-%m-%d')
        st.metric("First Seen", first_seen)

    with col3:
        last_updated = pd.to_datetime(patient['last_updated_at']).strftime('%Y-%m-%d %H:%M')
        st.metric("Last Updated", last_updated)

    st.markdown("---")

    # Risk Assessment
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Risk Assessment")

        if latest_strat:
            # Risk level badge
            show_risk_badge(latest_strat['risk_level'])

            # ML score gauge
            st.markdown("#### ML Risk Score")
            show_ml_score_gauge(latest_strat['ml_score'])

            # Override info
            if latest_strat['override_applied']:
                st.warning(f"""
                **⚠️ Rule Override Applied**

                Threshold-based level: {latest_strat['threshold_based_level']}

                Override reason: {latest_strat['override_reason']}
                """)
            else:
                st.info(f"Threshold-based assessment: {latest_strat['threshold_based_level']}")

        else:
            st.info("No risk stratification available for this patient.")

    with col2:
        st.markdown("### Recommendations")

        if latest_strat and latest_strat.get('recommendations'):
            show_recommendations_list(latest_strat['recommendations'])
        else:
            st.info("No recommendations available.")

    st.markdown("---")

    # ML Prediction Details
    if latest_pred:
        st.markdown("### 🤖 ML Prediction Details")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Class Probabilities")

            probabilities = latest_pred['probabilities']
            prob_df = pd.DataFrame({
                'Risk Class': probabilities.keys(),
                'Probability': [f"{v:.1%}" for v in probabilities.values()],
                'Raw Score': list(probabilities.values())
            })

            st.dataframe(prob_df[['Risk Class', 'Probability']], use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### Prediction Metadata")

            st.write(f"**Predicted Label:** {latest_pred['predicted_label']}")
            st.write(f"**ML Risk Score:** {latest_pred['ml_score']:.3f}")

            created_at = pd.to_datetime(latest_pred['created_at']).strftime('%Y-%m-%d %H:%M')
            st.write(f"**Prediction Date:** {created_at}")

        # Feature values (if available)
        if latest_pred.get('feature_values'):
            st.markdown("#### Feature Values Used for Prediction")

            feature_values = latest_pred['feature_values']

            # Display as table
            feature_df = pd.DataFrame([
                {'Feature': k, 'Value': f"{v:.2f}"} for k, v in feature_values.items()
            ])

            st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # SHAP Explanations (mock data for now - would come from explainer in full implementation)
    st.markdown("### 🔍 Top Contributing Features (SHAP)")

    st.info("""
    **Feature Importance Explanation:**

    These features had the most significant impact on the ML risk prediction.
    Features that INCREASE risk are shown in red, while features that DECREASE risk are shown in green.

    Note: SHAP values are generated during model training. Run the full pipeline to see actual explanations.
    """)

    # Mock SHAP data for demonstration
    if latest_pred and latest_pred.get('feature_values'):
        mock_top_features = []

        feature_values = latest_pred['feature_values']

        # Create mock explanations for top 3 features
        feature_display_names = {
            'resting_hr_estimate': 'Resting Heart Rate',
            'sleep_hours_avg': 'Average Sleep Duration',
            'activity_score': 'Physical Activity Score',
            'sedentary_ratio': 'Sedentary Behavior Ratio',
            'workout_consistency': 'Workout Consistency',
            'hr_variability_proxy': 'Heart Rate Variability',
            'mood_stress_ratio': 'Stress Indicator'
        }

        feature_units = {
            'resting_hr_estimate': 'bpm',
            'sleep_hours_avg': 'hours',
            'activity_score': 'points',
            'sedentary_ratio': 'ratio',
            'workout_consistency': 'ratio',
            'hr_variability_proxy': 'bpm std',
            'mood_stress_ratio': 'ratio'
        }

        # Take first 3 features as mock top contributors
        for feature_name in list(feature_values.keys())[:3]:
            mock_top_features.append({
                'feature_name': feature_name,
                'feature_display_name': feature_display_names.get(feature_name, feature_name),
                'value': feature_values[feature_name],
                'unit': feature_units.get(feature_name, ''),
                'impact': 'INCREASES' if latest_pred['ml_score'] > 0.5 else 'DECREASES',
                'abs_shap_value': 0.25 - len(mock_top_features) * 0.05  # Mock SHAP values
            })

        # Display feature explanations
        col1, col2, col3 = st.columns(3)

        for i, feature in enumerate(mock_top_features):
            with [col1, col2, col3][i]:
                show_feature_explanation(feature)

        # Feature importance chart
        if len(mock_top_features) > 0:
            plot_feature_importance(mock_top_features)

    st.markdown("---")

    # Prediction History
    st.markdown("### 📈 Prediction History")

    pred_history = cache.get_prediction_history(selected_patient_id, limit=10)

    if len(pred_history) > 0:
        history_df = pred_history[['created_at', 'ml_score', 'predicted_label']].copy()
        history_df['created_at'] = pd.to_datetime(history_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        history_df['ml_score'] = history_df['ml_score'].round(3)

        history_df.columns = ['Date', 'ML Score', 'Predicted Label']

        st.dataframe(history_df, use_container_width=True, hide_index=True)

        # Show trend
        if len(pred_history) > 1:
            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=pd.to_datetime(pred_history['created_at']),
                y=pred_history['ml_score'],
                mode='lines+markers',
                name='ML Score',
                line=dict(color='#007bff', width=2),
                marker=dict(size=8)
            ))

            # Add threshold lines
            fig.add_hline(y=0.33, line_dash="dash", line_color="green", annotation_text="Low/Medium threshold")
            fig.add_hline(y=0.66, line_dash="dash", line_color="red", annotation_text="Medium/High threshold")

            fig.update_layout(
                title="ML Score Trend",
                xaxis_title="Date",
                yaxis_title="ML Risk Score",
                yaxis_range=[0, 1],
                template='plotly_white',
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No prediction history available.")

    st.markdown("---")

    # Recent Observations
    st.markdown("### 📊 Recent Observations")

    observations = cache.get_patient_observations(selected_patient_id, limit=10)

    if len(observations) > 0:
        obs_df = observations[['observation_date', 'metric_name', 'value', 'unit']].copy()
        obs_df.columns = ['Date', 'Metric', 'Value', 'Unit']

        st.dataframe(obs_df, use_container_width=True, hide_index=True)
    else:
        st.info("No observations available.")

except Exception as e:
    st.error(f"Error loading risk dashboard: {e}")

    import traceback
    with st.expander("Error Details"):
        st.code(traceback.format_exc())

show_footer()
