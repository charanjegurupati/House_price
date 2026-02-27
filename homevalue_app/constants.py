from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = APP_ROOT / "model.pkl"
USERS_JSON_PATH = APP_ROOT / "users.json"
USERS_DB_PATH = APP_ROOT / "users.db"
SESSION_STORE_PATH = APP_ROOT / "auth_sessions.json"
LOGO_PATH = APP_ROOT / "assets" / "logo.svg"
ABOUT_IMG_1 = APP_ROOT / "assets" / "about_home.svg"
ABOUT_IMG_2 = APP_ROOT / "assets" / "about_ai.svg"
