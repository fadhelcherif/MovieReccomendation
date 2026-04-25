

 
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_predict_valid():
    response = client.post("/predict", json={"favorites": ["The Godfather"], "top_n": 3})
    assert response.status_code == 200
    assert "recommendations" in response.json()


def test_predict_invalid():
    response = client.post("/predict", json={"favorites": ["Nonexistent Movie"], "top_n": 3})
    assert response.status_code == 404

