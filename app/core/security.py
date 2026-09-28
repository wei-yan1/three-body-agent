"""Password hashing and small JWT helpers for local auth."""

from __future__ import annotations

import datetime as dt

import bcrypt
import jwt
from app.core.config.settings import get_jwt_secret, get_runtime_settings


JWT_ALGORITHM = "HS256"
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user_id: int, username: str) -> str:
    expire_minutes = int(get_runtime_settings()["jwt_expire_minutes"])
    now = dt.datetime.now(dt.UTC)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(minutes=expire_minutes)).timestamp()),
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
