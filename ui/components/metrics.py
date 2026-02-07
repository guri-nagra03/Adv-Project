"""
Metrics Display Components

Reusable components for displaying metrics and KPIs.
"""

import streamlit as st
from typing import Optional


def show_risk_badge(risk_level: str, size: str = "large"):
    """
    Display risk level badge with color and icon.

    Args:
        risk_level: Risk level ('Green', 'Yellow', 'Red')
        size: Badge size ('small', 'large')

    Example:
        >>> show_risk_badge('Red')
    """
    colors = {
        "Green": "#28a745",
        "Yellow": "#ffc107",
        "Red": "#dc3545"
    }

    icons = {
        "Green": "✓",
        "Yellow": "⚠",
        "Red": "🔴"
    }

    color = colors.get(risk_level, "#6c757d")
    icon = icons.get(risk_level, "?")

    if size == "large":
        st.markdown(
            f"""
            <div style="
                background-color: {color};
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            ">
                {icon} {risk_level} Risk
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <span style="
                background-color: {color};
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            ">
                {icon} {risk_level}
            </span>
            """,
            unsafe_allow_html=True
        )


def show_ml_score_gauge(ml_score: float):
    """
    Display ML risk score as gauge/progress bar.

    Args:
        ml_score: ML score (0-1)

    Example:
        >>> show_ml_score_gauge(0.72)
    """
    # Determine color based on score
    if ml_score < 0.33:
        color = "#28a745"  # Green
        risk_text = "Low Risk"
    elif ml_score < 0.66:
        color = "#ffc107"  # Yellow
        risk_text = "Medium Risk"
    else:
        color = "#dc3545"  # Red
        risk_text = "High Risk"

    percentage = int(ml_score * 100)

    st.markdown(f"**ML Risk Score: {ml_score:.3f}** ({risk_text})")

    st.markdown(
        f"""
        <div style="
            width: 100%;
            background-color: #f0f0f0;
            border-radius: 10px;
            height: 30px;
            position: relative;
            margin: 10px 0;
        ">
            <div style="
                width: {percentage}%;
                background-color: {color};
                border-radius: 10px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
            ">
                {percentage}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_metric_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    icon: Optional[str] = None
):
    """
    Display a metric card.

    Args:
        title: Metric title
        value: Metric value
        delta: Optional change indicator
        icon: Optional emoji icon

    Example:
        >>> show_metric_card("Total Patients", "156", "+12", "👥")
    """
    icon_html = f"{icon} " if icon else ""

    delta_html = ""
    if delta:
        delta_color = "#28a745" if delta.startswith("+") else "#dc3545"
        delta_html = f'<div style="color: {delta_color}; font-size: 14px;">{delta}</div>'

    st.markdown(
        f"""
        <div style="
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #007bff;
        ">
            <div style="color: #6c757d; font-size: 14px;">{icon_html}{title}</div>
            <div style="font-size: 28px; font-weight: bold; margin: 5px 0;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def show_recommendations_list(recommendations: list):
    """
    Display recommendations as formatted list.

    Args:
        recommendations: List of recommendation strings

    Example:
        >>> show_recommendations_list([
        ...     "Increase physical activity",
        ...     "Improve sleep hygiene"
        ... ])
    """
    st.markdown("### 📋 Recommendations")

    for i, rec in enumerate(recommendations, 1):
        # Check if recommendation has special prefix
        if rec.startswith("⚠️"):
            st.error(rec)
        else:
            st.markdown(f"{i}. {rec}")


def show_feature_explanation(feature: dict):
    """
    Display SHAP feature explanation.

    Args:
        feature: Feature dictionary with explanation

    Example:
        >>> feature = {
        ...     'feature_display_name': 'Resting Heart Rate',
        ...     'value': 95,
        ...     'unit': 'bpm',
        ...     'impact': 'INCREASES',
        ...     'explanation': 'Resting Heart Rate (95 bpm) - INCREASES risk...'
        ... }
        >>> show_feature_explanation(feature)
    """
    impact_color = "#dc3545" if feature.get('impact') == 'INCREASES' else "#28a745"
    impact_icon = "⬆" if feature.get('impact') == 'INCREASES' else "⬇"

    st.markdown(
        f"""
        <div style="
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid {impact_color};
        ">
            <div style="font-weight: bold; color: {impact_color};">
                {impact_icon} {feature.get('feature_display_name', feature.get('feature_name'))}
            </div>
            <div style="font-size: 20px; margin: 5px 0;">
                {feature.get('value'):.1f} {feature.get('unit', '')}
            </div>
            <div style="font-size: 12px; color: #6c757d;">
                {feature.get('impact', 'AFFECTS').title()} risk
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
