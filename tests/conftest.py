from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import create_app


@pytest.fixture(autouse=True)
def disable_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_API_KEY", raising=False)


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
