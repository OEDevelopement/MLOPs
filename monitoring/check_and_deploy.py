import json
import mlflow
import os
import subprocess

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
client = mlflow.tracking.MlflowClient()

# Lade Validierungsergebnis
with open("monitoring/validation_result.json", "r") as f:
    validation_status = json.load(f).get("status")

if validation_status != "pass":
    print("Modell hat Validierung nicht bestanden. Deployment abgebrochen.")
    exit(1)

# Lade Metriken des neuen Modells
latest_run = client.search_runs(order_by=["start_time desc"], max_results=1)[0]
new_metrics = latest_run.data.metrics

# Lade Metriken des aktuellen Produktionsmodells
production_model = client.get_latest_versions("production_model", stages=["Production"])
if production_model:
    production_run = client.get_run(production_model[0].run_id)
    production_metrics = production_run.data.metrics
else:
    print("Kein produktives Modell gefunden. Setze das neue Modell direkt als Produktion!")
    better_model = True

# Vergleich: Ist das neue Modell besser?
better_model = False
if production_model:
    better_model = new_metrics["accuracy"] > production_metrics["accuracy"]

if better_model:
    print("Neues Modell ist besser. Setze es als Produktionsmodell.")
    client.transition_model_version_stage(
        name="production_model",
        version=latest_run.info.version,
        stage="Production"
    )

    print("Neues Modell wird als Docker-Image gebaut und deployed...")
    subprocess.run([
        "docker", "build", "-t", f"myrepo/model:{latest_run.info.version}", "-f", "mlflow/Dockerfile", "."
    ])
    subprocess.run([
        "docker", "push", f"myrepo/model:{latest_run.info.version}"
    ])
else:
    print("Neues Modell ist schlechter als das Produktionsmodell. Kein Deployment.")
    exit(1)
