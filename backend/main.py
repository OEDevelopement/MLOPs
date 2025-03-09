from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
import json
import os
import time
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# Model service Endpoint konfigurieren
MODEL_SERVICE_URL = os.environ.get('MODEL_SERVICE_URL', 'http://model_service:8080/invocations')

class Person(BaseModel):
    age: int
    workclass: str
    educational_num: int = Field(alias="educational-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    hours_per_week: int = Field(alias="hours-per-week")
    is_Male: int  # Maintaining exact case from DataFrame
    income_over_50K: int = Field(alias="income >50K")
    is_White: int  # Maintaining exact case from DataFrame
    from_USA: int  # Maintaining exact case from DataFrame
    gained_capital: int = Field(alias="gained-capital")
    
    class Config:
        populate_by_name = True


app = FastAPI()

# Custom Prometheus metrics
PREDICTION_COUNTER = Counter(
    'income_predictions_total', 
    'Total number of income predictions made',
    ['result']
)

MODEL_REQUEST_TIME = Histogram(
    'model_request_duration_seconds',
    'Histogram of model service request duration in seconds',
    buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Set up Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

### Post data from frontend to MODEL SERVICE 

@app.post("/predict")
async def predict(input_person: Person):
    """
    Empfängt JSON-Daten vom Client, sendet diese an den Model Service und gibt die Vorhersage zurück.
    """
    # Track model service request time
    with MODEL_REQUEST_TIME.time():
        try:
            # Sende eine POST-Anfrage an den Model-Service-Endpoint mit den Inputdaten
            model_dict = input_person.model_dump(by_alias=True)
            split_format = {
                "columns": list(model_dict.keys()),
                "data": [list(model_dict.values())],
                "index": [0]
            }
            headers = {"Content-Type": "application/json"}

            # Use the manually created split format dictionary as the payload
            payload = json.dumps({"dataframe_split": split_format})

            response = requests.post(MODEL_SERVICE_URL, data=payload, headers=headers)
            print(f"Model Service response: {response.status_code}, {response.text}")
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # Fehlerbehandlung, falls die Anfrage fehlschlägt
            raise HTTPException(status_code=500, detail=f"Fehler bei der Vorhersage-Anfrage: {e}")
    
    # Get prediction result and increment counter
    result = response.json()
    prediction_value = result.get('predictions', [0])[0]
    outcome = "over_50k" if prediction_value == 1 else "under_50k"
    PREDICTION_COUNTER.labels(result=outcome).inc()
    
    # Rückgabe der Vorhersage
    return result


@app.get("/")
async def root():
    return {"message": "Backend API is running"}


@app.get("/health")
async def health_check():
    """
    Basic health check endpoint that can be used by Prometheus for uptime monitoring
    """
    return {"status": "healthy", "service": "income-prediction-backend"}