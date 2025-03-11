import pandas as pd
import itertools
import mlflow
import mlflow.sklearn
import subprocess
import shutil
import datetime
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from param_grid_functions import is_valid_lr_params, select_diverse_combinations

# --------------------------------------
# mlflow parameter setzen
# --------------------------------------

# Verwende die Umgebungsvariable oder einen Fallback-Wert
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# MLflow Experiment erstellen
experiment_name = "income_prediction"
experiment = mlflow.set_experiment(experiment_name)

# --------------------------------------
# Daten erstellen
# --------------------------------------

data = pd.read_csv('data/processed/processed_data.csv')

X = data.drop(columns=['income >50K'])
y = data['income >50K']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------
# preprocessor für die pipeline erstellen
# --------------------------------------

# Numerische und kategoriale Spalten identifizieren
num_features = ['age', 'educational-num', 'hours-per-week']
cat_features = ['workclass', 'marital-status', 'occupation', 'relationship']

# Transformationen für numerische Spalten (Skalierung)
num_transformer = StandardScaler()

# Transformationen für kategoriale Spalten (One-Hot-Encoding)
cat_transformer = OneHotEncoder(handle_unknown='ignore')

# Preprocessing-Pipeline
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# --------------------------------------
# Hyperparaneter Kombinationen für die drei Modelle erstellen
# --------------------------------------

# **Hyperparameter-Sets definieren**
param_grid_rf = {
    "n_estimators": [100, 200],  
    "max_depth": [5, 10],        
    "min_samples_split": [2, 5], 
    "min_samples_leaf": [1, 3]   
}

param_combinations_rf = list(itertools.product(*param_grid_rf.values()))

param_grid_lr = {
    "penalty": ["l1", "l2", "elasticnet", None],  
    "C": [0.01, 0.1, 1, 10, 100],  
    "solver": ["liblinear", "lbfgs", "saga"],  
    "max_iter": [100, 200, 500]  
}

param_combinations_lr = list(itertools.product(*param_grid_lr.values()))

param_grid_gb = {
    "n_estimators": [100, 200],  
    "learning_rate": [0.01, 0.1, 0.2],  
    "max_depth": [3, 5, 10],  
    "min_samples_split": [2, 5],  
    "min_samples_leaf": [1, 3]  
}

param_combinations_gb = list(itertools.product(*param_grid_gb.values()))

# **Kompatible Kombinationen filtern**
param_combinations_lr = [params for params in param_combinations_lr if is_valid_lr_params(params, param_grid_lr)]

# **Maximal 20 Kombinationen pro Modell**
param_combinations_rf = select_diverse_combinations(param_combinations_rf, 10)
param_combinations_lr = select_diverse_combinations(param_combinations_lr, 10)
param_combinations_gb = select_diverse_combinations(param_combinations_gb, 10)

# Modelle mit ihren Parametern durchlaufen
model_configs = [
    ("random_forest", RandomForestClassifier, param_combinations_rf, param_grid_rf),
    ("logistic_regression", LogisticRegression, param_combinations_lr, param_grid_lr),
    ("gradient_boosting", GradientBoostingClassifier, param_combinations_gb, param_grid_gb)
]

# --------------------------------------
# mlflow run durchführen und tracken
# --------------------------------------

for model_type, model_class, param_combinations, param_grid in model_configs:
    for params in param_combinations:
        # Dictionary mit aktuellen Parametern
        current_params = dict(zip(param_grid.keys(), params))

        with mlflow.start_run():
            print(f"Training {model_type} mit Parametern: {current_params}")

            # Modell initialisieren
            model = model_class(**current_params, random_state=42) if "random_state" in model_class().get_params() else model_class(**current_params)

            # Pipeline mit Preprocessing und Modell
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            # Modell trainieren
            pipeline.fit(X_train, y_train)

            # Vorhersagen & Accuracy berechnen
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Parameter & Modelltyp loggen
            mlflow.log_params(current_params)
            mlflow.log_param("model_type", model_type)

            # Metriken loggen
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)

            # Modell speichern
            mlflow.sklearn.log_model(pipeline, f"{model_type}_pipeline")

# ...
# Rest des Codes bleibt gleich bis zu diesem Teil:

# --------------------------------------
# bestes Modell raussuchen und speichern
# --------------------------------------

# Modell-Ausgabepfad aus der Umgebungsvariable holen
model_output_path = os.environ.get("MODEL_OUTPUT_PATH", "/app/best_model")

# Prüfen, ob bereits ein Modell existiert
if os.path.exists(model_output_path) and os.listdir(model_output_path):
    print(f"Vorhandenes Modell in {model_output_path} gefunden.")
    
    # Archiv-Verzeichnis erstellen (im übergeordneten Ordner)
    parent_dir = os.path.dirname(model_output_path)
    archive_dir = os.path.join(parent_dir, "archived_models")
    os.makedirs(archive_dir, exist_ok=True)
    
    # Zeitstempel für eindeutigen Archivnamen
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_output_path)
    archive_path = os.path.join(archive_dir, f"{model_name}_{timestamp}")
    
    # Altes Modell ins Archiv verschieben
    print(f"Archiviere vorhandenes Modell nach {archive_path}...")
    shutil.move(model_output_path, archive_path)
    
    # Neues leeres Verzeichnis erstellen
    os.makedirs(model_output_path, exist_ok=True)
    print(f"Neues Verzeichnis {model_output_path} für aktuelles Modell erstellt.")
else:
    # Verzeichnis erstellen, falls es nicht existiert
    os.makedirs(model_output_path, exist_ok=True)

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
best_run = runs.loc[runs["metrics.f1_score"].idxmax()]
model_type = best_run["params.model_type"]
run_id = best_run['run_id']
model_uri = f"runs:/{run_id}/{model_type}_pipeline"

print(f"Speichere bestes Modell ({model_type}) in {model_output_path}...")
mlflow.sklearn.save_model(
        sk_model=mlflow.sklearn.load_model(model_uri),
        path=model_output_path
    )
print(f"Modell erfolgreich gespeichert!")
print(f"F1-Score des besten Modells: {best_run['metrics.f1_score']:.4f}")