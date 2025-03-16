import json
import pytest
from unittest.mock import patch, MagicMock
import requests

# Import your FastAPI app
from backend.main import app, Person

# Create TestClient using FastAPI's built-in testing tools
from fastapi.testclient import TestClient

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint returns expected message"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_person_model_validation():
    """Test Person model validates input correctly"""
    # Valid person data
    valid_data = {
        "age": 30,
        "workclass": "Private", 
        "educational-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "hours-per-week": 40,
        "is_Male": 1,
        "income >50K": 1,
        "is_White": 1,
        "from_USA": 1,
        "gained-capital": 0
    }
    
    # Create Person instance
    person = Person(**valid_data)
    
    # Check conversions happened correctly
    assert person.age == 30
    assert person.educational_num == 13
    assert person.marital_status == "Married-civ-spouse"
    assert person.hours_per_week == 40
    assert person.income_over_50K == 1
    assert person.gained_capital == 0

@patch('backend.main.requests.post')
def test_predict_endpoint_success(mock_post):
    """Test predict endpoint with successful MLflow response"""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"predictions": [1]}
    mock_response.text = '{"predictions": [1]}'
    mock_post.return_value = mock_response
    
    # Test data
    test_data = {
        "age": 30,
        "workclass": "Private", 
        "educational-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "hours-per-week": 40,
        "is_Male": 1,
        "income >50K": 1,
        "is_White": 1,
        "from_USA": 1,
        "gained-capital": 0
    }
    
    # Make request to the endpoint
    response = client.post("/predict", json=test_data)
    
    # Check response
    assert response.status_code == 200
    assert response.json() == {"predictions": [1]}
    
    # Verify the mock was called with expected payload
    called_payload = json.loads(mock_post.call_args[1]['data'])
    assert "dataframe_split" in called_payload
    assert set(called_payload["dataframe_split"]["columns"]) == set(test_data.keys())
    assert len(called_payload["dataframe_split"]["data"]) == 1

@patch('backend.main.requests.post')
def test_predict_endpoint_mlflow_error(mock_post):
    """Test predict endpoint handling MLflow server errors"""
    # Setup mock to raise an exception
    mock_post.side_effect = requests.exceptions.RequestException("MLflow server error")
    
    # Test data
    test_data = {
        "age": 30,
        "workclass": "Private", 
        "educational-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "hours-per-week": 40,
        "is_Male": 1,
        "income >50K": 1,
        "is_White": 1,
        "from_USA": 1,
        "gained-capital": 0
    }
    
    # Make request to the endpoint
    response = client.post("/predict", json=test_data)
    
    # Check that the endpoint returns a 500 error
    assert response.status_code == 500


def test_predict_endpoint_invalid_input():
    """Test predict endpoint with invalid input data"""
    # Missing required fields
    invalid_data = {
        "age": 30,
        "workclass": "Private"
        # Missing other required fields
    }
    
    # Make request to the endpoint
    response = client.post("/predict", json=invalid_data)
    
    # Check that the endpoint returns a validation error
    assert response.status_code == 422  # Unprocessable Entity