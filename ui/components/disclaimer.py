"""
Educational Disclaimer Component

Displays prominent disclaimer on all pages.
"""

import streamlit as st


def show_disclaimer():
    """
    Display educational disclaimer banner.

    Call this at the top of every page.

    Example:
        >>> show_disclaimer()
    """
    st.warning(
        "⚠️ **EDUCATIONAL DEMONSTRATION ONLY** – "
        "This application does NOT provide medical advice, diagnosis, or treatment. "
        "For medical concerns, consult a qualified healthcare provider."
    )


def show_detailed_disclaimer():
    """
    Display detailed disclaimer in expander.

    Use on main pages for comprehensive disclaimer.
    """
    with st.expander("⚠️ Important Disclaimer - Please Read"):
        st.markdown("""
        ### Educational Demonstration Only

        **CardioGuard is a student-level health informatics project designed for
        educational purposes ONLY.**

        #### This Application Does NOT:
        - Provide medical advice, diagnosis, or treatment
        - Replace professional medical consultation
        - Detect or diagnose heart disease or any medical condition
        - Provide emergency medical services
        - Make clinical decisions

        #### This Application DOES:
        - Demonstrate HL7 FHIR interoperability standards
        - Showcase machine learning interpretability (SHAP)
        - Illustrate preventive wellness monitoring concepts
        - Use synthetic risk labels for demonstration

        #### Data & Privacy:
        - Uses synthetic/demonstration data only
        - No real patient data or medical records
        - Local-only processing (no external data sharing)
        - Educational disclaimer included in all FHIR resources

        #### For Medical Concerns:
        - Consult a qualified healthcare provider
        - Call emergency services (911) for emergencies
        - Use validated medical devices and clinical systems

        ---

        *CardioGuard is a demonstration of health informatics concepts
        and should not be used for any clinical or medical purposes.*
        """)


def show_footer():
    """
    Display page footer with disclaimer reminder.

    Example:
        >>> show_footer()
    """
    st.markdown("---")
    st.caption(
        "CardioGuard v1.0 | Educational Health Informatics Demonstration | "
        "NOT for clinical use"
    )
