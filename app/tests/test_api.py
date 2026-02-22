import pytest 
from fastapi.testclient import TestClient
from app.main import app 
client = TestClient(app)

def test_api_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

#want to write a test to basically make sure that the model has a silhoute score of 70 or higher
