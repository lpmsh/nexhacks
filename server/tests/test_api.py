import requests

LOCALHOST = "http://localhost:8000"

def test_health_check():
    response = requests.get(f"{LOCALHOST}/health")
    assert response.status_code == 200
    
