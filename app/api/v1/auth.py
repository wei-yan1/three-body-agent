"""Auth API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import create_access_token, decode_access_token, hash_password, verify_password
from app.schemas.auth import AuthRequest, AuthResponse, UserOut
from app.storage.mysql.client import mysql_connection
from app.storage.mysql.schema import init_mysql_schema


router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
bearer = HTTPBearer(auto_error=False)


def _user_out(row: dict) -> UserOut:
    return UserOut(
        id=int(row["id"]),
        username=str(row["username"]),
        display_name=row.get("display_name"),
        avatar_url=row.get("avatar_url"),
    )


@router.post("/register", response_model=AuthResponse)
def register(payload: AuthRequest) -> AuthResponse:
    init_mysql_schema()
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="用户名不能为空")

    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
            if cursor.fetchone():
                raise HTTPException(status_code=409, detail="用户名已存在")
            cursor.execute(
                """
                INSERT INTO users (username, password_hash, display_name)
                VALUES (%s, %s, %s)
                """,
                (username, hash_password(payload.password), username),
            )
            user_id = int(cursor.lastrowid)
            cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
            user = cursor.fetchone()

    token = create_access_token(user_id=user_id, username=username)
    return AuthResponse(access_token=token, user=_user_out(user))


@router.post("/login", response_model=AuthResponse)
def login(payload: AuthRequest) -> AuthResponse:
    init_mysql_schema()
    username = payload.username.strip()
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
            user = cursor.fetchone()
            if not user or not verify_password(payload.password, str(user["password_hash"])):
                raise HTTPException(status_code=401, detail="用户名或密码错误")
            if user.get("status") != "active":
                raise HTTPException(status_code=403, detail="账号不可用")
            cursor.execute("UPDATE users SET last_login_at=NOW() WHERE id=%s", (user["id"],))

    token = create_access_token(user_id=int(user["id"]), username=str(user["username"]))
    return AuthResponse(access_token=token, user=_user_out(user))


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> UserOut:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = int(payload["sub"])
    except Exception as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已失效") from error

    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
            user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return _user_out(user)


@router.get("/me", response_model=UserOut)
def me(user: UserOut = Depends(current_user)) -> UserOut:
    return user
