import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score

mlflow.set_tracking_uri("http://host.docker.internal:5001")
# Experiment laden
experiment_name = "income_prediction"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    raise ValueError(f"Experiment '{experiment_name}' nicht gefunden!")

experiment_id = experiment.experiment_id

# Alle Runs abrufen
runs = mlflow.search_runs(experiment_ids=[experiment_id])

# Besten Run anhand des F1-Scores finden
best_run = runs.loc[runs["metrics.f1_score"].idxmax()]
best_run_id = best_run["run_id"]
model_type = best_run["params.model_type"]

print(f"Bestes Modell: {model_type}, Run ID: {best_run_id}")

# Modell aus MLflow laden
model_uri = f"runs:/{best_run_id}/{model_type}_pipeline"
model = mlflow.sklearn.load_model(model_uri)

# Testdaten erneut laden
data = pd.read_csv("data/processed/processed_data.csv")
X = data.drop(columns=["income >50K"])
y = data["income >50K"]

# 🔹 1. ROC-Kurve berechnen
y_probs = model.predict_proba(X)[:, 1]  # Wahrscheinlichkeit für Klasse 1
fpr, tpr, _ = roc_curve(y, y_probs)
roc_auc = auc(fpr, tpr)

# ROC-Kurve plotten & speichern
plt.figure()
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {model_type}")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

# 🔹 2. Kreuzvalidierung durchführen
cv_f1_scores = cross_val_score(model, X, y, cv=5, scoring="f1_weighted")
cv_accuracy_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
cv_auc_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

# Durchschnittswerte berechnen
mean_f1 = cv_f1_scores.mean()
mean_accuracy = cv_accuracy_scores.mean()
mean_auc = cv_auc_scores.mean()

# 🔹 3. In den besten Run loggen
with mlflow.start_run(run_id=best_run_id):
    # ROC-Kurve als Artifact loggen
    mlflow.log_artifact("roc_curve.png")

    # Kreuzvalidierungsergebnisse als Metriken loggen
    mlflow.log_metric("cv_f1_score", mean_f1)
    mlflow.log_metric("cv_accuracy", mean_accuracy)
    mlflow.log_metric("cv_auc", mean_auc)

os.remove("roc_curve.png")

print(f"ROC-Kurve und Kreuzvalidierungsmetriken wurden in den Run {best_run_id} geloggt!")
