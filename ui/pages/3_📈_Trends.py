"""
Trends Page

Time-series analysis and visualization of wearable data.
"""

import streamlit as st
import pandas as pd

from src.auth.simple_auth import require_authentication
from src.storage.sqlite_cache import SQLiteCache
from ui.components.disclaimer import show_disclaimer, show_footer
from ui.components.charts import (
    plot_time_series,
    plot_rolling_average,
    plot_scatter
)


# Page configuration
st.set_page_config(
    page_title="Trends - CardioGuard",
    page_icon="📈",
    layout="wide"
)

# Require authentication
require_authentication()

# Header
st.title("📈 Trends Analysis")
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
    selected_patient_id = st.selectbox(
        "Select Patient",
        options=patients_df['patient_id'].tolist(),
        index=0,
        format_func=lambda x: f"Patient {x}"
    )

    # Load observations
    observations = cache.get_patient_observations(selected_patient_id, limit=1000)

    if len(observations) == 0:
        st.warning(f"No observations found for Patient {selected_patient_id}")
        st.stop()

    st.markdown(f"**Analyzing {len(observations)} observations for Patient {selected_patient_id}**")

    st.markdown("---")

    # Metrics available
    available_metrics = observations['metric_name'].unique().tolist()

    if len(available_metrics) == 0:
        st.info("No metrics available for this patient.")
        st.stop()

    # Time range selection
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📅 Time Range")
        observations['observation_date'] = pd.to_datetime(observations['observation_date'])

        min_date = observations['observation_date'].min()
        max_date = observations['observation_date'].max()

        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_obs = observations[
                (observations['observation_date'] >= pd.Timestamp(start_date)) &
                (observations['observation_date'] <= pd.Timestamp(end_date))
            ]
        else:
            filtered_obs = observations

    with col2:
        st.markdown("### 📊 Available Metrics")
        st.write(f"Total metrics: {len(available_metrics)}")

        for metric in available_metrics[:5]:
            count = (filtered_obs['metric_name'] == metric).sum()
            st.write(f"- {metric}: {count} obs")

    with col3:
        st.markdown("### 🎯 Summary")
        st.metric("Total Observations", len(filtered_obs))
        st.metric("Date Range", f"{(filtered_obs['observation_date'].max() - filtered_obs['observation_date'].min()).days} days")

    st.markdown("---")

    # Metric selection for detailed view
    st.markdown("### 📊 Metric Trends")

    selected_metric = st.selectbox(
        "Select Metric to Visualize",
        options=available_metrics,
        format_func=lambda x: x.replace('_', ' ').title()
    )

    # Filter by selected metric
    metric_data = filtered_obs[filtered_obs['metric_name'] == selected_metric].copy()
    metric_data = metric_data.sort_values('observation_date')

    if len(metric_data) == 0:
        st.info(f"No data for {selected_metric}")
    else:
        # Metric info
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Observations", len(metric_data))

        with col2:
            st.metric("Mean", f"{metric_data['value'].mean():.1f} {metric_data['unit'].iloc[0]}")

        with col3:
            st.metric("Min", f"{metric_data['value'].min():.1f} {metric_data['unit'].iloc[0]}")

        with col4:
            st.metric("Max", f"{metric_data['value'].max():.1f} {metric_data['unit'].iloc[0]}")

        # Time series plot
        st.markdown(f"#### {selected_metric.replace('_', ' ').title()} Over Time")

        # Healthy ranges (could be loaded from config)
        healthy_ranges = {
            'heart_rate_avg': (60, 80),
            'sleep_hours': (7, 9)
        }

        healthy_range = healthy_ranges.get(selected_metric)

        plot_time_series(
            df=metric_data,
            x_col='observation_date',
            y_col='value',
            title=f"{selected_metric.replace('_', ' ').title()} Trend",
            y_label=f"{selected_metric.replace('_', ' ').title()} ({metric_data['unit'].iloc[0]})",
            healthy_range=healthy_range
        )

        # Rolling average (if enough data)
        if len(metric_data) >= 7:
            st.markdown("#### 7-Day Rolling Average")

            plot_rolling_average(
                df=metric_data,
                x_col='observation_date',
                y_col='value',
                title=f"{selected_metric.replace('_', ' ').title()} with 7-Day Average",
                window=7,
                y_label=f"{selected_metric.replace('_', ' ').title()} ({metric_data['unit'].iloc[0]})"
            )

    st.markdown("---")

    # Multi-metric comparison
    st.markdown("### 📊 Multi-Metric Comparison")

    if len(available_metrics) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            metric1 = st.selectbox(
                "Select First Metric",
                options=available_metrics,
                key="metric1"
            )

        with col2:
            metric2 = st.selectbox(
                "Select Second Metric",
                options=[m for m in available_metrics if m != metric1],
                key="metric2"
            )

        # Get data for both metrics
        metric1_data = filtered_obs[filtered_obs['metric_name'] == metric1].copy()
        metric2_data = filtered_obs[filtered_obs['metric_name'] == metric2].copy()

        # Merge on date
        merged_data = pd.merge(
            metric1_data[['observation_date', 'value']],
            metric2_data[['observation_date', 'value']],
            on='observation_date',
            suffixes=('_1', '_2')
        )

        if len(merged_data) > 0:
            st.markdown(f"#### {metric1.replace('_', ' ').title()} vs {metric2.replace('_', ' ').title()}")

            plot_scatter(
                df=merged_data,
                x_col='value_1',
                y_col='value_2',
                title=f"Correlation: {metric1.replace('_', ' ').title()} vs {metric2.replace('_', ' ').title()}",
                x_label=metric1.replace('_', ' ').title(),
                y_label=metric2.replace('_', ' ').title()
            )

            # Correlation
            corr = merged_data['value_1'].corr(merged_data['value_2'])
            st.metric("Correlation Coefficient", f"{corr:.3f}")

            if abs(corr) > 0.7:
                st.success(f"Strong {'positive' if corr > 0 else 'negative'} correlation detected")
            elif abs(corr) > 0.4:
                st.info(f"Moderate {'positive' if corr > 0 else 'negative'} correlation detected")
            else:
                st.info("Weak or no correlation detected")
        else:
            st.info("No overlapping dates for these two metrics")

    else:
        st.info("Need at least 2 metrics for comparison")

    st.markdown("---")

    # All metrics overview
    st.markdown("### 📊 All Metrics Summary")

    summary_data = []

    for metric in available_metrics:
        metric_subset = filtered_obs[filtered_obs['metric_name'] == metric]

        if len(metric_subset) > 0:
            summary_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Count': len(metric_subset),
                'Mean': f"{metric_subset['value'].mean():.1f}",
                'Min': f"{metric_subset['value'].min():.1f}",
                'Max': f"{metric_subset['value'].max():.1f}",
                'Unit': metric_subset['unit'].iloc[0]
            })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Error loading trends: {e}")

    import traceback
    with st.expander("Error Details"):
        st.code(traceback.format_exc())

show_footer()
