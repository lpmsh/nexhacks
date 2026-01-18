
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

import requests
import pytest


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200

    assert response.json()["status"] == "healthy"



@pytest.mark.parametrize("market", ["NASDAQ", "NYSE", "DOWJONES"])
def test_create_market(market):
    r = requests.post(f"{LOCALHOST}/new/market", json={"market_name": market})
    assert r.status_code == 200
    print(f"Create Market Response: {r.json()}")

