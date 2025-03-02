import great_expectations as ge
import json
import mlflow
import os

# Verbindung zu MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("Model Validation")

# Lade Metriken des neuen Modells aus MLflow
client = mlflow.tracking.MlflowClient()
latest_run = client.search_runs(order_by=["start_time desc"], max_results=1)[0]
new_metrics = latest_run.data.metrics

# Erwartete Mindestwerte
expected_metrics = {
    "accuracy": 0.85,
    "f1_score": 0.80,
    "rmse": 0.5  # Je nach Modell anpassen
}

# Great Expectations Validierung
context = ge.DataContext()

results = []
for metric, min_value in expected_metrics.items():
    result = new_metrics.get(metric, 0) >= min_value
    results.append(result)
    print(f"Validating {metric}: {new_metrics.get(metric, 0)} >= {min_value} → {result}")

if all(results):
    print("Neues Modell hat die Qualitätsanforderungen bestanden.")
    with open("monitoring/validation_result.json", "w") as f:
        json.dump({"status": "pass"}, f)
else:
    print("Neues Modell hat die Qualitätsanforderungen nicht bestanden!")
    with open("monitoring/validation_result.json", "w") as f:
        json.dump({"status": "fail"}, f)
    exit(1)  # Beendet das Skript mit Fehlerstatus, um die Pipeline zu stoppen
