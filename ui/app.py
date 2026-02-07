"""
CardioGuard Main Application

Streamlit dashboard for cardiovascular wellness monitoring.

IMPORTANT: Educational demonstration only - NOT for clinical use.
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from src.auth.simple_auth import require_authentication, logout
from ui.components.disclaimer import show_disclaimer, show_footer


# Page configuration
st.set_page_config(
    page_title="CardioGuard - Cardiovascular Wellness Monitoring",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Require authentication
require_authentication()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #007bff;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 👤 User Info")
    user_info = st.session_state.get("user_info", {})
    st.write(f"**{user_info.get('name', 'User')}**")
    st.write(f"Role: {user_info.get('role', 'N/A').title()}")

    st.markdown("---")

    st.markdown("### 📊 Navigation")
    st.markdown("""
    Use the navigation above to access:
    - **Patient List** - View all patients
    - **Risk Dashboard** - Individual patient analysis
    - **Trends** - Time-series analysis
    - **FHIR Export** - View FHIR resources
    """)

    st.markdown("---")

    # Logout button
    if st.button("🚪 Logout", use_container_width=True):
        logout()

    st.markdown("---")
    st.caption("CardioGuard v1.0")
    st.caption("Educational Demo Only")

# Main content
st.markdown('<div class="main-header">❤️ CardioGuard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cardiovascular Wellness Monitoring System</div>', unsafe_allow_html=True)

# Educational disclaimer
show_disclaimer()

st.markdown("---")

# Welcome content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Welcome to CardioGuard")

    st.markdown("""
    CardioGuard is a **student-level health informatics demonstration** showcasing:

    - **HL7 FHIR R4 Interoperability** - Standards-compliant health data exchange
    - **Machine Learning with Explainability** - SHAP-based interpretable predictions
    - **Risk Stratification** - Educational wellness risk categorization
    - **Wearable Data Integration** - Processing fitness tracker metrics

    #### Key Features:

    - 📊 **Patient List** - Browse all patients with risk levels
    - 🎯 **Risk Dashboard** - Detailed patient risk analysis with ML explanations
    - 📈 **Trends** - Visualize temporal patterns in wearable data
    - 🔄 **FHIR Export** - View and download FHIR-compliant resources

    #### Getting Started:

    1. Navigate to **Patient List** to see all patients
    2. Select a patient to view their **Risk Dashboard**
    3. Explore **Trends** for time-series analysis
    4. Check **FHIR Export** to see interoperability in action
    """)

with col2:
    st.markdown("### 📊 System Overview")

    # Load cache stats
    try:
        from src.storage.sqlite_cache import SQLiteCache

        cache = SQLiteCache()
        stats = cache.get_stats()

        st.metric("Total Patients", stats.get('total_patients', 0), help="Number of patients in system")

        risk_dist = stats.get('risk_distribution', {})
        st.metric("Green Risk", risk_dist.get('Green', 0), help="Low risk patients")
        st.metric("Yellow Risk", risk_dist.get('Yellow', 0), help="Medium risk patients")
        st.metric("Red Risk", risk_dist.get('Red', 0), help="High risk patients")

        st.metric("Total Observations", stats.get('total_observations', 0), help="FHIR Observations cached")

    except Exception as e:
        st.info("No data loaded yet. Please run the training script first.")

st.markdown("---")

# Important notes
st.markdown("### ⚠️ Important Notes")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **Educational Only**

    This is a demonstration system for learning health informatics concepts.
    NOT intended for clinical use.
    """)

with col2:
    st.info("""
    **Synthetic Data**

    All risk labels are generated using rule-based logic.
    No real patient outcomes data.
    """)

with col3:
    st.info("""
    **Standards Compliant**

    Uses HL7 FHIR R4 with proper LOINC codes for interoperability demonstration.
    """)

st.markdown("---")

# Architecture overview
with st.expander("🏗️ System Architecture"):
    st.markdown("""
    ### CardioGuard Architecture

    **Data Pipeline:**
    - Fitness tracker CSV → Preprocessing → Feature Engineering → ML Model

    **ML Pipeline:**
    - Logistic Regression with synthetic labels
    - SHAP explanations for interpretability
    - Risk stratification (Green/Yellow/Red)

    **FHIR Integration:**
    - Wearable data → FHIR Observations (LOINC codes)
    - ML predictions → RiskAssessment resources
    - High/medium risk → Flag resources
    - HAPI FHIR server for storage

    **Storage:**
    - SQLite cache for fast queries
    - FHIR server for standards compliance
    - Graceful degradation if FHIR unavailable

    **UI:**
    - Streamlit dashboard
    - Authentication (demo credentials)
    - 4 main pages with visualizations
    """)

show_footer()
