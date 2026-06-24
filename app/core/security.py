"""Password hashing and small JWT helpers for local auth."""

from __future__ import annotations

import datetime as dt
import os

import bcrypt
import jwt


JWT_ALGORITHM = "HS256"


def get_jwt_secret() -> str:
    return os.getenv("JWT_SECRET_KEY", "dev-three-body-agent-secret")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user_id: int, username: str) -> str:
    expire_minutes = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))
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
