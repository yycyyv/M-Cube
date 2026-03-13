from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

# Ensure repository root is importable in CI (for `from main import create_app`).
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import create_app


@pytest.fixture(autouse=True)
def disable_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_API_KEY", raising=False)


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
