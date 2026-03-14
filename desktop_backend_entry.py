from __future__ import annotations

import os

import uvicorn
from main import app as fastapi_app


def main() -> None:
    host = os.getenv("MCUBE_BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("MCUBE_BACKEND_PORT", "8000"))
    # Use direct app object import so PyInstaller reliably bundles backend entry deps.
    uvicorn.run(fastapi_app, host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
