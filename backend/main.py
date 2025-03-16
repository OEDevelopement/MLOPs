from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import requests
import json
import os
import time
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

MLFLOW_URL = os.environ.get('MODEL_SERVICE_URL','http://localhost:8080/invocations')
API_URL = os.environ.get('API_URL','http://backend:8000/predict')

# Simple Prometheus metrics
API_CALLS = Counter('api_calls_total', 'Total number of API calls', ['endpoint'])
PREDICTIONS = Counter('predictions_total', 'Total number of predictions made')
PREDICTION_ERRORS = Counter('prediction_errors_total', 'Total number of prediction errors')
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')

class Person(BaseModel):
    age: int
    workclass: str
    educational_num: int = Field(alias="educational-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    hours_per_week: int = Field(alias="hours-per-week")
    is_Male: int
    is_White: int
    from_USA: int
    gained_capital: int = Field(alias="gained-capital")
    
    class Config:
        populate_by_name = True

app = FastAPI()

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    API_CALLS.labels(endpoint=request.url.path).inc()
    
    try:
        response = await call_next(request)
        return response
    finally:
        ACTIVE_REQUESTS.dec()

@app.get("/metrics")
async def metrics():
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
async def predict(input_person: Person):
    """
    Empfängt JSON-Daten vom Client, sendet diese an den MLflow Server und gibt die Vorhersage zurück.
    """
    try:
        # Sende eine POST-Anfrage an den MLflow-Endpoint mit den Inputdaten
        model_dict = input_person.model_dump(by_alias=True)
        split_format = {
            "columns": list(model_dict.keys()),
            "data": [list(model_dict.values())],
            "index": [0]
        }
        headers = {"Content-Type": "application/json"}

        # Use the manually created split format dictionary as the payload
        payload = json.dumps({"dataframe_split": split_format})

        response = requests.post(MLFLOW_URL, data=payload, headers=headers)
        print(response.status_code, response.text)
        response.raise_for_status()
        
        # Increment prediction counter
        PREDICTIONS.inc()
    except requests.exceptions.RequestException as e:
        # Increment error counter and raise exception
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"Fehler bei der Vorhersage-Anfrage: {e}")
    
    # Rückgabe der MLflow-Vorhersage
    return response.json()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "ok"}