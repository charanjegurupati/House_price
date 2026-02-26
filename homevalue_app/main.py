import streamlit as st

from .auth import (
    init_session,
    render_login,
    render_sidebar,
    render_signup,
    restore_auth_from_query,
    show_center_welcome,
)
from .styles import inject_styles
from .views import render_about, render_brand, render_home


def main() -> None:
    st.set_page_config(page_title="HomeValue AI", page_icon="🏠", layout="wide")

    init_session()
    restore_auth_from_query()
    inject_styles()
    render_sidebar()
    render_brand()

    if st.session_state["authenticated"] and st.session_state.get("show_welcome"):
        show_center_welcome(st.session_state["user"].get("name", "User"))
        return

    if st.session_state["authenticated"]:
        render_home()
        return

    view = st.session_state.get("view", "About")
    if view == "Login":
        render_login()
    elif view == "Sign Up":
        render_signup()
    else:
        render_about()
