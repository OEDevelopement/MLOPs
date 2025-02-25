from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

MLFLOW_URL = os.environ.get('MLFLOW_TRACKING_URI','localhost:5000')

class person(BaseModel):
    age: int
    workclass: str
    educational_num: int
    martial_status: str
    occupation: str
    relationship: str
    race: str
    # add further or change depending on frontend

app = FastAPI()


### Post data from frontend to MLFLOW Server 

@app.post("/predict")
async def predict(input_data: person):
    """
    Empfängt JSON-Daten vom Client, sendet diese an den MLflow Server und gibt die Vorhersage zurück.
    """
    try:
        # Sende eine POST-Anfrage an den MLflow-Endpoint mit den Inputdaten
        response = requests.post(MLFLOW_URL, json=input_data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # Fehlerbehandlung, falls die Anfrage fehlschlägt
        raise HTTPException(status_code=500, detail=f"Fehler bei der Vorhersage-Anfrage: {e}")
    
    # Rückgabe der MLflow-Vorhersage
    return response.json()


@app.get("/")
async def root():
    return {"message": "Hello World"}