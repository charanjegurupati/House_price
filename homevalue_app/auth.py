from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from uuid import uuid4

import streamlit as st

from .constants import SESSION_STORE_PATH, USERS_PATH

SESSION_TTL_SECONDS = 60 * 60 * 24 * 30


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email))


def load_users() -> dict[str, dict]:
    if not USERS_PATH.exists():
        return {}
    try:
        return json.loads(USERS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_users(users: dict[str, dict]) -> None:
    USERS_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")


def load_sessions() -> dict[str, dict]:
    if not SESSION_STORE_PATH.exists():
        return {}
    try:
        return json.loads(SESSION_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_sessions(sessions: dict[str, dict]) -> None:
    SESSION_STORE_PATH.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


def cleanup_sessions(sessions: dict[str, dict]) -> dict[str, dict]:
    now = time.time()
    return {
        sid: info
        for sid, info in sessions.items()
        if float(info.get("expires_at", 0)) > now and str(info.get("username", "")).strip()
    }


def get_query_sid() -> str:
    sid = st.query_params.get("sid")
    if isinstance(sid, list):
        return str(sid[0]) if sid else ""
    return str(sid) if sid else ""


def create_session(username: str) -> str:
    sessions = cleanup_sessions(load_sessions())
    sid = uuid4().hex
    sessions[sid] = {
        "username": username,
        "expires_at": time.time() + SESSION_TTL_SECONDS,
    }
    save_sessions(sessions)
    return sid


def remove_session(sid: str) -> None:
    if not sid:
        return
    sessions = cleanup_sessions(load_sessions())
    if sid in sessions:
        sessions.pop(sid, None)
        save_sessions(sessions)


def resolve_session_user(sid: str) -> str:
    if not sid:
        return ""
    sessions = cleanup_sessions(load_sessions())
    info = sessions.get(sid)
    save_sessions(sessions)
    if not info:
        return ""
    return str(info.get("username", "")).strip()


def restore_auth_from_query() -> None:
    if st.session_state.get("authenticated"):
        return

    sid = get_query_sid()
    if not sid:
        return

    username = resolve_session_user(sid)
    if not username:
        st.query_params.clear()
        return

    users = load_users()
    profile = users.get(username)
    if not profile:
        remove_session(sid)
        st.query_params.clear()
        return

    st.session_state["authenticated"] = True
    st.session_state["user"] = {
        "username": username,
        "name": profile.get("name", username),
        "email": profile.get("email", ""),
    }
    st.session_state["view"] = "Home"


def init_session() -> None:
    defaults = {
        "authenticated": False,
        "user": {},
        "view": "Login",
        "flash": "",
        "show_welcome": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    st.sidebar.markdown("### Navigation")
    current_view = st.session_state.get("view", "About")

    if st.sidebar.button(
        "🔐 Login",
        use_container_width=True,
        type="primary" if current_view == "Login" else "secondary",
    ):
        st.session_state["view"] = "Login"
        st.rerun()
    if st.sidebar.button(
        "📝 Sign Up",
        use_container_width=True,
        type="primary" if current_view == "Sign Up" else "secondary",
    ):
        st.session_state["view"] = "Sign Up"
        st.rerun()
    if st.sidebar.button(
        "ℹ️ About",
        use_container_width=True,
        type="primary" if current_view == "About" else "secondary",
    ):
        st.session_state["view"] = "About"
        st.rerun()

    st.sidebar.markdown("---")
    if st.session_state["authenticated"]:
        st.sidebar.success(f"Logged in as {st.session_state['user'].get('name', 'User')}")
        if st.sidebar.button("Logout", use_container_width=True):
            sid = get_query_sid()
            remove_session(sid)
            st.query_params.clear()
            st.session_state["authenticated"] = False
            st.session_state["user"] = {}
            st.session_state["show_welcome"] = False
            st.session_state["view"] = "Login"
            st.session_state.pop("last_input_values", None)
            st.session_state.pop("last_prediction", None)
            st.rerun()
    else:
        st.sidebar.info("Login is required to run predictions.")


def render_login() -> None:
    st.markdown(
        """
        <div class="hero">
            <h3 style="margin:0;">Login</h3>
            <p style="margin:6px 0 0 0;">Access prediction and analytics features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("flash"):
        st.success(st.session_state["flash"])
        st.session_state["flash"] = ""

    with st.form("login_form", clear_on_submit=False):
        identity = st.text_input("Username or Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", type="primary")

    if not submit:
        return

    users = load_users()
    ident = identity.strip().lower()
    matched: tuple[str, dict] | None = None
    for uname, profile in users.items():
        if ident == uname.lower() or ident == str(profile.get("email", "")).lower():
            matched = (uname, profile)
            break

    if not matched:
        st.error("User not found.")
        return

    uname, profile = matched
    if hash_password(password) != profile.get("password_hash", ""):
        st.error("Incorrect password.")
        return

    st.session_state["authenticated"] = True
    st.session_state["user"] = {
        "username": uname,
        "name": profile.get("name", uname),
        "email": profile.get("email", ""),
    }
    sid = create_session(uname)
    st.query_params["sid"] = sid
    st.session_state["show_welcome"] = True
    st.session_state["view"] = "Home"
    st.rerun()


def render_signup() -> None:
    st.markdown(
        """
        <div class="hero">
            <h3 style="margin:0;">Sign Up</h3>
            <p style="margin:6px 0 0 0;">Create an account to enable prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("signup_form", clear_on_submit=True):
        name = st.text_input("Full Name")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Create Account")

    if not submit:
        return

    users = load_users()
    uname = username.strip().lower()
    clean_name = name.strip()
    clean_email = email.strip().lower()

    if not clean_name:
        st.error("Name is required.")
        return
    if not uname:
        st.error("Username is required.")
        return
    if uname in users:
        st.error("Username already exists.")
        return
    if not valid_email(clean_email):
        st.error("Enter a valid email.")
        return
    if any(str(u.get("email", "")).lower() == clean_email for u in users.values()):
        st.error("Email already used.")
        return
    if len(password) < 6:
        st.error("Password must be at least 6 characters.")
        return
    if password != confirm_password:
        st.error("Passwords do not match.")
        return

    users[uname] = {
        "name": clean_name,
        "email": clean_email,
        "password_hash": hash_password(password),
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    save_users(users)
    st.session_state["view"] = "Login"
    st.session_state["flash"] = "Registration successful. Please login."
    st.rerun()


def show_center_welcome(name: str) -> None:
    safe_name = str(name).replace("<", "").replace(">", "")
    message = f"Welcome, {safe_name}"
    placeholder = st.empty()

    for i in range(1, len(message) + 1):
        placeholder.markdown(
            f'<div class="welcome-center">{message[:i]}<span class="cursor">|</span></div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.045)

    time.sleep(0.7)
    placeholder.empty()
    st.session_state["show_welcome"] = False
    st.rerun()
