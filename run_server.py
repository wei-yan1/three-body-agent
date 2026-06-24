"""Start the local FastAPI development server on port 1314."""

from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=1314,
        reload=True,
    )
