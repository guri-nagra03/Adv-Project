"""
Simple Authentication

Basic authentication for CardioGuard demo application.

IMPORTANT: This is a demonstration-only authentication system.
DO NOT use in production. Passwords are hardcoded.

Demo Users:
- clinician1 / demo123
- admin / admin456
"""

import hashlib
from typing import Optional, Dict
import streamlit as st

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


# Demo users (DEMO ONLY - DO NOT USE IN PRODUCTION)
DEMO_USERS = {
    "clinician1": {
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "role": "clinician",
        "name": "Dr. Demo Clinician"
    },
    "admin": {
        "password_hash": hashlib.sha256("admin456".encode()).hexdigest(),
        "role": "admin",
        "name": "Admin User"
    }
}


def authenticate(username: str, password: str) -> bool:
    """
    Authenticate user with username and password.

    Args:
        username: Username
        password: Password (plain text)

    Returns:
        True if authentication successful, False otherwise

    Example:
        >>> if authenticate("clinician1", "demo123"):
        ...     print("Login successful")
    """
    if username not in DEMO_USERS:
        logger.warning(f"Login attempt with unknown username: {username}")
        return False

    user = DEMO_USERS[username]
    password_hash = hashlib.sha256(password.encode()).hexdigest()

    if password_hash == user["password_hash"]:
        logger.info(f"Successful login: {username}")
        return True
    else:
        logger.warning(f"Failed login attempt: {username}")
        return False


def get_user_info(username: str) -> Optional[Dict]:
    """
    Get user information.

    Args:
        username: Username

    Returns:
        User info dictionary or None
    """
    if username in DEMO_USERS:
        return {
            "username": username,
            "role": DEMO_USERS[username]["role"],
            "name": DEMO_USERS[username]["name"]
        }
    return None


def check_authentication() -> bool:
    """
    Check if user is authenticated in Streamlit session.

    Returns:
        True if authenticated, False otherwise

    Usage in Streamlit:
        >>> if not check_authentication():
        ...     st.stop()  # Halt execution if not authenticated
    """
    return st.session_state.get("authenticated", False)


def login_page():
    """
    Display Streamlit login page.

    Call this function to show the login form.
    Sets st.session_state['authenticated'] to True on successful login.

    Example:
        >>> if not check_authentication():
        ...     login_page()
        ...     st.stop()
    """
    st.set_page_config(
        page_title="CardioGuard - Login",
        page_icon="❤️",
        layout="wide"
    )

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.title("❤️ CardioGuard")
        st.subheader("Cardiovascular Wellness Monitoring")

        # Educational disclaimer
        st.warning(
            "⚠️ **EDUCATIONAL DEMONSTRATION ONLY**\n\n"
            "This application does NOT provide medical advice, diagnosis, or treatment. "
            "For medical concerns, consult a qualified healthcare provider."
        )

        st.markdown("---")

        # Login form
        with st.form("login_form"):
            st.subheader("🔐 Clinician Login")

            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")

            submit = st.form_submit_button("Login", use_container_width=True)

            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                elif authenticate(username, password):
                    # Set session state
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.session_state["user_info"] = get_user_info(username)

                    st.success(f"Welcome, {st.session_state['user_info']['name']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        # Demo credentials
        st.markdown("---")
        with st.expander("📝 Demo Credentials"):
            st.markdown("""
            **Clinician Account:**
            - Username: `clinician1`
            - Password: `demo123`

            **Admin Account:**
            - Username: `admin`
            - Password: `admin456`

            *Note: This is a demonstration system with hardcoded credentials.*
            """)

        # Footer
        st.markdown("---")
        st.caption("CardioGuard v1.0 | Educational Health Informatics Demonstration")


def logout():
    """
    Logout current user.

    Clears authentication session state.
    """
    if "authenticated" in st.session_state:
        username = st.session_state.get("username", "unknown")
        logger.info(f"User logged out: {username}")

        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["user_info"] = None

    st.rerun()


def require_authentication():
    """
    Decorator/helper to require authentication.

    Use at the top of each page to ensure user is logged in.

    Example:
        >>> require_authentication()
        >>> # Rest of page code here
    """
    if not check_authentication():
        login_page()
        st.stop()


# Example usage
if __name__ == "__main__":
    print("=== Simple Authentication Demo ===\n")

    # Test authentication
    print("1. Testing valid credentials:")
    if authenticate("clinician1", "demo123"):
        print("   ✓ clinician1 authenticated successfully")

    print("\n2. Testing invalid credentials:")
    if not authenticate("clinician1", "wrongpassword"):
        print("   ✓ Invalid password rejected correctly")

    print("\n3. Testing unknown user:")
    if not authenticate("unknown_user", "password"):
        print("   ✓ Unknown user rejected correctly")

    print("\n4. Getting user info:")
    user_info = get_user_info("clinician1")
    print(f"   Username: {user_info['username']}")
    print(f"   Role: {user_info['role']}")
    print(f"   Name: {user_info['name']}")

    print("\n5. Demo users:")
    for username, user_data in DEMO_USERS.items():
        print(f"   - {username}: {user_data['name']} ({user_data['role']})")
