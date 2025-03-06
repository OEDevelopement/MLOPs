from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
import requests
import json
import os

MLFLOW_URL = os.environ.get('MLFLOW_SERVING_URI','http://localhost:5000')

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


### Post data from frontend to MLFLOW Server 

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

        response = requests.post(f"{MLFLOW_URL}/invocations", data=payload, headers=headers)
        print(response.status_code, response.text)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # Fehlerbehandlung, falls die Anfrage fehlschlägt
        raise HTTPException(status_code=500, detail=f"Fehler bei der Vorhersage-Anfrage: {e}")
    
    # Rückgabe der MLflow-Vorhersage
    return response.json()


@app.get("/")
async def root():
    return {"message": "Hello World"}