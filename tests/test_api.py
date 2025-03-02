#Api-Endpunkt-Tests
# BEISPIEL-INPUT

from fastapi.testclient import TestClient
from frontend.app import app  # Importiert die FastAPI-App

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={"input_data": [0.5, 1.2, 3.4]})
    assert response.status_code == 200
    assert "prediction" in response.json()
