"""FastAPI application entrypoint."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "frontend/static/assets"
LOGIN_BACKGROUND = ASSET_DIR / "login-background.jpg"
LUOJI_IMAGE = ASSET_DIR / "luoji.jpg"
ZHANG_BEIHAI_IMAGE = ASSET_DIR / "zbh.jpg"
ZHANG_BEIHAI_CHAT_IMAGE = ASSET_DIR / "zhang.jpg"
WANGMIAO_IMAGE = ASSET_DIR / "wangmiao.png"
YEWENJIE_IMAGE = ASSET_DIR / "yewenjie.jpg"
GLOBAL_MUSIC = ASSET_DIR / "global-music.mp3"

load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(title="Three Body Temporal Persona Agent")
app.include_router(auth_router)
app.include_router(chat_router)
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "frontend/static"), name="static")


@app.get("/")
def login_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend/login.html")


@app.get("/agents")
def agents_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend/agents.html")


@app.get("/chat/luoji")
def luoji_chat_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend/luoji-chat.html")


@app.get("/chat/zhangbeihai")
def zhangbeihai_chat_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend/zhangbeihai-chat.html")


@app.get("/chat/wangmiao")
def wangmiao_chat_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend/wangmiao-chat.html")


@app.get("/chat/yewenjie")
def yewenjie_chat_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend/yewenjie-chat.html")


@app.get("/assets/login-background")
def login_background() -> FileResponse:
    if not LOGIN_BACKGROUND.exists():
        raise HTTPException(status_code=404, detail="Login background image not found")
    return FileResponse(LOGIN_BACKGROUND)


@app.get("/assets/luoji")
def luoji_image() -> FileResponse:
    if not LUOJI_IMAGE.exists():
        raise HTTPException(status_code=404, detail="Luo Ji image not found")
    return FileResponse(LUOJI_IMAGE)


@app.get("/assets/zhangbeihai")
def zhangbeihai_image() -> FileResponse:
    if not ZHANG_BEIHAI_IMAGE.exists():
        raise HTTPException(status_code=404, detail="Zhang Beihai image not found")
    return FileResponse(ZHANG_BEIHAI_IMAGE)


@app.get("/assets/zhangbeihai-chat")
def zhangbeihai_chat_image() -> FileResponse:
    if not ZHANG_BEIHAI_CHAT_IMAGE.exists():
        raise HTTPException(status_code=404, detail="Zhang Beihai chat image not found")
    return FileResponse(ZHANG_BEIHAI_CHAT_IMAGE)


@app.get("/assets/wangmiao")
def wangmiao_image() -> FileResponse:
    if not WANGMIAO_IMAGE.exists():
        raise HTTPException(status_code=404, detail="Wang Miao image not found")
    return FileResponse(WANGMIAO_IMAGE)


@app.get("/assets/yewenjie")
def yewenjie_image() -> FileResponse:
    if not YEWENJIE_IMAGE.exists():
        raise HTTPException(status_code=404, detail="Ye Wenjie image not found")
    return FileResponse(YEWENJIE_IMAGE)


@app.get("/assets/global-music")
def global_music() -> FileResponse:
    if not GLOBAL_MUSIC.exists():
        raise HTTPException(status_code=404, detail="Global music file not found")
    return FileResponse(GLOBAL_MUSIC, media_type="audio/mpeg")
