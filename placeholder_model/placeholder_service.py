import os
import time
import mlflow
import mlflow.pyfunc
from flask import Flask, request, jsonify
import requests
import json
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# MLflow Konfiguration
MLFLOW_URI = os.environ.get("MLFLOW_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "income_prediction")
model = None

def load_best_model():
    """Lädt das beste Modell aus MLflow"""
    global model
    
    logger.info(f"Verbinde mit MLflow unter {MLFLOW_URI}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    try:
        # Experiment abrufen
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            logger.warning(f"Experiment '{EXPERIMENT_NAME}' nicht gefunden. Warte...")
            return False
            
        # Alle Runs abrufen
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if runs.empty:
            logger.warning("Keine Runs gefunden. Warte auf Training...")
            return False
            
        # Besten Run finden
        best_run = runs.loc[runs["metrics.f1_score"].idxmax()]
        best_run_id = best_run["run_id"]
        model_type = best_run["params.model_type"]
        
        logger.info(f"Bestes Modell gefunden: {model_type}, Run ID: {best_run_id}")
        
        # Modell laden
        model_uri = f"runs:/{best_run_id}/{model_type}_pipeline"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Modell erfolgreich geladen!")
        return True
        
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells: {e}")
        return False

@app.route('/health')
def health():
    """Health Check Endpoint"""
    if model is None:
        return jsonify({
            "status": "initializing",
            "message": "Modell wird geladen"
        }), 503
    else:
        return jsonify({
            "status": "ready",
            "message": "Model Service bereit"
        })

@app.route('/invocations', methods=['POST'])
def invocations():
    """Inference Endpoint"""
    # Wenn kein Modell geladen ist, versuche es erneut zu laden
    if model is None:
        success = load_best_model()
        if not success:
            return jsonify({
                "predictions": [0],
                "message": "Modell wird noch trainiert. Bitte später erneut versuchen."
            }), 503
    
    # Daten aus der Anfrage extrahieren
    request_data = request.get_json(silent=True)
    
    try:
        # Die Daten für die Vorhersage vorbereiten
        if 'dataframe_split' in request_data:
            data = request_data['dataframe_split']
            # Modell für Vorhersage verwenden
            predictions = model.predict(data)
            return jsonify({"predictions": predictions.tolist()})
        else:
            return jsonify({
                "error": "Ungültiges Datenformat. 'dataframe_split' erwartet."
            }), 400
            
    except Exception as e:
        logger.error(f"Fehler bei der Vorhersage: {e}")
        return jsonify({
            "error": f"Fehler bei der Vorhersage: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Beim Start versuchen, das Modell zu laden
    # Wenn es nicht gelingt, startet der Service trotzdem und versucht es später erneut
    logger.info("Model Service startet...")
    
    # Warte auf MLflow-Server
    max_retries = 12
    retry_delay = 10
    
    for i in range(max_retries):
        try:
            requests.get(f"{MLFLOW_URI}/api/2.0/mlflow/experiments/list")
            logger.info("MLflow-Server ist bereit!")
            break
        except requests.exceptions.ConnectionError:
            logger.info(f"Warte auf MLflow-Server... {i+1}/{max_retries}")
            time.sleep(retry_delay)
    
    # Versuche, das Modell zu laden
    load_best_model()
    
    # Start des Flask-Servers
    app.run(host='0.0.0.0', port=8080)