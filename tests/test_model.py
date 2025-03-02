# Modelltests
# BEISPIEL-INPUT

import pytest
import mlflow.pyfunc
import numpy as np

@pytest.fixture
def example_input():
    return np.array([[0.5, 1.2, 3.4]])

def test_model_output(example_input):
    model = mlflow.pyfunc.load_model("models:/mein-modell/latest")
    prediction = model.predict(example_input)
    
    assert prediction is not None
    assert isinstance(prediction, np.ndarray)
