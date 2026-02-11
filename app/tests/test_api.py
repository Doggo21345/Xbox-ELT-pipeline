import pytest 
from fastapi.testclient import TestClient
from app.main import app 
client = TestClient(app)

def test_api_status():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

#want to write a test to basically make sure that the model has a silhoute score of 70 or higher
