import requests
import pytest

LOCALHOST = "http://localhost:8000"


def test_health_check():
    response = requests.get(f"{LOCALHOST}/health")
    assert response.status_code == 200


@pytest.mark.parametrize("market", ["NASDAQ", "NYSE", "DOWJONES"])
def test_create_market(market):
    r = requests.post(f"{LOCALHOST}/new/market", json={"market_name": market})
    assert r.status_code == 200
    print(f"Create Market Response: {r.json()}")