import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_redirects_to_app():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/app/"


def test_examples_endpoint():
    response = client.get("/examples")
    assert response.status_code == 200
    data = response.json()
    assert "examples" in data
    assert isinstance(data["examples"], list)
    assert len(data["examples"]) > 0


def test_compress_endpoint():
    response = client.post(
        "/compress",
        json={
            "text": "This is a test document with some content that needs compression.",
            "query": "What is this about?",
            "target_ratio": 0.5,
            "run_baselines": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "compressed_text" in data
    assert "metrics" in data
    assert "spans" in data
