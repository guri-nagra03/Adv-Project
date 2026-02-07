"""
Chart Components

Reusable visualization components using Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Optional


def plot_time_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_label: str = None,
    color: str = "#007bff",
    healthy_range: tuple = None
):
    """
    Plot time series line chart.

    Args:
        df: DataFrame with time series data
        x_col: Column name for x-axis (date)
        y_col: Column name for y-axis (metric)
        title: Chart title
        y_label: Y-axis label
        color: Line color
        healthy_range: Optional (min, max) tuple for healthy range shading

    Example:
        >>> plot_time_series(
        ...     df=data,
        ...     x_col='date',
        ...     y_col='heart_rate_avg',
        ...     title='Heart Rate Trend',
        ...     y_label='Heart Rate (bpm)',
        ...     healthy_range=(60, 80)
        ... )
    """
    fig = go.Figure()

    # Add healthy range if provided
    if healthy_range:
        fig.add_hrect(
            y0=healthy_range[0],
            y1=healthy_range[1],
            fillcolor="lightgreen",
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text="Healthy Range",
            annotation_position="top left"
        )

    # Add main line
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='lines+markers',
        name=y_label or y_col,
        line=dict(color=color, width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label or y_col,
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_rolling_average(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    window: int = 7,
    y_label: str = None
):
    """
    Plot time series with rolling average.

    Args:
        df: DataFrame with time series data
        x_col: Column name for x-axis (date)
        y_col: Column name for y-axis (metric)
        title: Chart title
        window: Rolling window size (days)
        y_label: Y-axis label

    Example:
        >>> plot_rolling_average(
        ...     df=data,
        ...     x_col='date',
        ...     y_col='steps',
        ...     title='Daily Steps with 7-Day Average',
        ...     window=7
        ... )
    """
    fig = go.Figure()

    # Raw data
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        name='Daily',
        marker=dict(color='lightblue', size=4),
        opacity=0.5
    ))

    # Rolling average
    rolling_avg = df[y_col].rolling(window=window, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=rolling_avg,
        mode='lines',
        name=f'{window}-Day Average',
        line=dict(color='#007bff', width=3)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label or y_col,
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_risk_distribution(risk_counts: dict):
    """
    Plot risk level distribution pie chart.

    Args:
        risk_counts: Dictionary of {risk_level: count}

    Example:
        >>> plot_risk_distribution({'Green': 80, 'Yellow': 15, 'Red': 5})
    """
    colors = {
        'Green': '#28a745',
        'Yellow': '#ffc107',
        'Red': '#dc3545'
    }

    labels = list(risk_counts.keys())
    values = list(risk_counts.values())
    chart_colors = [colors.get(label, '#6c757d') for label in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=chart_colors),
        textinfo='label+percent+value',
        hovertemplate='%{label}<br>%{value} patients<br>%{percent}<extra></extra>'
    )])

    fig.update_layout(
        title="Risk Level Distribution",
        template='plotly_white',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance(top_features: list):
    """
    Plot SHAP feature importance bar chart.

    Args:
        top_features: List of feature dictionaries with 'feature_display_name' and 'abs_shap_value'

    Example:
        >>> features = [
        ...     {'feature_display_name': 'Resting HR', 'abs_shap_value': 0.25},
        ...     {'feature_display_name': 'Sleep Hours', 'abs_shap_value': 0.18}
        ... ]
        >>> plot_feature_importance(features)
    """
    # Sort by importance
    sorted_features = sorted(
        top_features,
        key=lambda x: x.get('abs_shap_value', 0),
        reverse=True
    )

    names = [f.get('feature_display_name', f.get('feature_name', ''))
             for f in sorted_features]
    values = [f.get('abs_shap_value', 0) for f in sorted_features]

    # Color by impact
    colors_list = ['#dc3545' if f.get('impact') == 'INCREASES' else '#28a745'
                   for f in sorted_features]

    fig = go.Figure(data=[go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker=dict(color=colors_list),
        text=[f"{v:.3f}" for v in values],
        textposition='auto',
    )])

    fig.update_layout(
        title="Top Contributing Features (SHAP Values)",
        xaxis_title="Absolute SHAP Value (Impact on Risk)",
        yaxis_title="Feature",
        template='plotly_white',
        height=300 + len(names) * 40,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str = None,
    title: str = None,
    x_label: str = None,
    y_label: str = None
):
    """
    Plot scatter plot.

    Args:
        df: DataFrame
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_col: Optional column for color coding
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label

    Example:
        >>> plot_scatter(
        ...     df=data,
        ...     x_col='mood_stress_ratio',
        ...     y_col='heart_rate_avg',
        ...     title='Stress vs Heart Rate'
        ... )
    """
    if color_col:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title or f"{y_col} vs {x_col}",
            labels={x_col: x_label or x_col, y_col: y_label or y_col},
            template='plotly_white'
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=title or f"{y_col} vs {x_col}",
            labels={x_col: x_label or x_col, y_col: y_label or y_col},
            template='plotly_white'
        )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def plot_histogram(
    df: pd.DataFrame,
    col: str,
    title: str = None,
    x_label: str = None,
    bins: int = 30
):
    """
    Plot histogram.

    Args:
        df: DataFrame
        col: Column to plot
        title: Chart title
        x_label: X-axis label
        bins: Number of bins

    Example:
        >>> plot_histogram(df, 'activity_score', 'Activity Score Distribution')
    """
    fig = px.histogram(
        df,
        x=col,
        nbins=bins,
        title=title or f"Distribution of {col}",
        labels={col: x_label or col},
        template='plotly_white'
    )

    fig.update_layout(
        xaxis_title=x_label or col,
        yaxis_title="Count",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)
