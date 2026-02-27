from __future__ import annotations

import json
import sqlite3
from typing import Any

from .constants import USERS_DB_PATH, USERS_JSON_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(USERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_store() -> None:
    USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            )
            """
        )
        conn.commit()

    _migrate_from_json_if_needed()


def _is_empty(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()
    return int(row["count"]) == 0


def _migrate_from_json_if_needed() -> None:
    if not USERS_JSON_PATH.exists():
        return

    raw = USERS_JSON_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return

    try:
        data = json.loads(raw)
    except Exception:
        return

    if not isinstance(data, dict):
        return

    with _connect() as conn:
        if not _is_empty(conn):
            return

        for username, profile in data.items():
            if not isinstance(profile, dict):
                continue
            uname = str(username).strip().lower()
            if not uname:
                continue
            name = str(profile.get("name", uname)).strip() or uname
            email = str(profile.get("email", "")).strip().lower()
            pwd_hash = str(profile.get("password_hash", "")).strip()
            created = str(profile.get("created_at_utc", "")).strip()
            if not (email and pwd_hash and created):
                continue

            conn.execute(
                """
                INSERT OR IGNORE INTO users (username, name, email, password_hash, created_at_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (uname, name, email, pwd_hash, created),
            )
        conn.commit()


def find_user_by_identity(identity: str) -> dict[str, Any] | None:
    ident = identity.strip().lower()
    if not ident:
        return None

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT username, name, email, password_hash, created_at_utc
            FROM users
            WHERE lower(username) = ? OR lower(email) = ?
            LIMIT 1
            """,
            (ident, ident),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_username(username: str) -> dict[str, Any] | None:
    uname = username.strip().lower()
    if not uname:
        return None
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT username, name, email, password_hash, created_at_utc
            FROM users
            WHERE lower(username) = ?
            LIMIT 1
            """,
            (uname,),
        ).fetchone()
    return dict(row) if row else None


def username_exists(username: str) -> bool:
    uname = username.strip().lower()
    if not uname:
        return False
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE lower(username) = ? LIMIT 1", (uname,)
        ).fetchone()
    return row is not None


def email_exists(email: str) -> bool:
    clean = email.strip().lower()
    if not clean:
        return False
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE lower(email) = ? LIMIT 1", (clean,)
        ).fetchone()
    return row is not None


def create_user(
    *,
    username: str,
    name: str,
    email: str,
    password_hash: str,
    created_at_utc: str,
) -> bool:
    uname = username.strip().lower()
    clean_name = name.strip()
    clean_email = email.strip().lower()
    pwd_hash = password_hash.strip()
    created = created_at_utc.strip()
    if not (uname and clean_name and clean_email and pwd_hash and created):
        return False

    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO users (username, name, email, password_hash, created_at_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (uname, clean_name, clean_email, pwd_hash, created),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
