import pandas as pd
import itertools
import os
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import subprocess

from param_grid_functions import is_valid_lr_params, select_diverse_combinations

# Check if training has already been completed
if os.path.exists("/mlflow/training_complete.flag"):
    print("Training has already been completed. Exiting...")
    exit(0)

# --------------------------------------
# mlflow parameter setzen
# --------------------------------------

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# MLflow Experiment erstellen
experiment_name = "income_prediction"
experiment = mlflow.set_experiment(experiment_name)

# --------------------------------------
# Daten erstellen
# --------------------------------------

data = pd.read_csv('/app/data/processed/processed_data.csv')

X = data.drop(columns=['income >50K'])
y = data['income >50K']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------
# preprocessor für die pipeline erstellen
# --------------------------------------

# Numerische und kategoriale Spalten identifizieren
num_features = ['age', 'educational-num', 'hours-per-week', 'gained-capital']
cat_features = ['workclass', 'marital-status', 'occupation', 'relationship']

# Transformationen für numerische Spalten (Skalierung)
num_transformer = StandardScaler()

# Transformationen für kategoriale Spalten (One-Hot-Encoding)
cat_transformer = OneHotEncoder(handle_unknown='ignore')

# Preprocessing-Pipeline
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
], remainder='passthrough')

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
param_combinations_rf = select_diverse_combinations(param_combinations_rf, 20)
param_combinations_lr = select_diverse_combinations(param_combinations_lr, 20)
param_combinations_gb = select_diverse_combinations(param_combinations_gb, 20)

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

# --------------------------------------
# bestes Modell raussuchen und als Docker-Image exportieren
# --------------------------------------

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
best_run = runs.loc[runs["metrics.f1_score"].idxmax()]

model_type = best_run["params.model_type"]
run_id = best_run['run_id']
model_uri = f"runs:/{run_id}/{model_type}_pipeline"

docker_image_name = "income-model-service"
docker_image_tag = run_id[:8]
docker_image = f"{docker_image_name}:{docker_image_tag}"

try:
    # Docker-Image mit MLflow erstellen
    print(f"Erstelle Docker-Image für Modell: {model_uri}")
    mlflow_build_command = [
        "mlflow", "models", "build-docker",
        "-m", model_uri,
        "-n", docker_image_name
    ]
    
    # Erzeuge das Docker-Image
    subprocess.run(mlflow_build_command, check=True)
    
    # Tagge zusätzlich mit dem Run-ID als Version
    tag_command = [
        "docker", "tag", 
        f"{docker_image_name}:latest", 
        docker_image
    ]
    subprocess.run(tag_command, check=True)
    
    print(f"MLflow Docker-Image erstellt: {docker_image_name}:latest und {docker_image}")
    
    # Informationen über das beste Modell speichern
    with open("/mlflow/current_model_info.txt", "w") as f:
        f.write(f"MODEL_URI={model_uri}\n")
        f.write(f"IMAGE_NAME={docker_image_name}\n")
        f.write(f"IMAGE_TAG={docker_image_tag}\n")
        f.write(f"FULL_IMAGE={docker_image}\n")
        f.write(f"RUN_ID={run_id}\n")
        f.write(f"ACCURACY={best_run['metrics.accuracy']}\n")
        f.write(f"F1_SCORE={best_run['metrics.f1_score']}\n")
        f.write(f"MODEL_TYPE={model_type}\n")
    
    # Training-Flag erstellen
    with open("/mlflow/training_complete.flag", "w") as f:
        f.write(f"Training completed at: {pd.Timestamp.now()}\n")
        f.write(f"Best model: {model_type}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"F1 Score: {best_run['metrics.f1_score']}\n")
        f.write(f"Docker image: {docker_image}\n")
    
    print("Model training completed. Best model information saved to /mlflow/current_model_info.txt")
    print("Training flag created at /mlflow/training_complete.flag")
    
except subprocess.CalledProcessError as e:
    print(f"Error building Docker image: {str(e)}")
    print("Continuing with saving model information...")
    
    # Selbst bei Fehler Modellinformation und Trainings-Flag speichern
    with open("/mlflow/current_model_info.txt", "w") as f:
        f.write(f"MODEL_URI={model_uri}\n")
        f.write(f"IMAGE_NAME={docker_image_name}\n")
        f.write(f"IMAGE_TAG={docker_image_tag}\n")
        f.write(f"RUN_ID={run_id}\n")
        f.write(f"ACCURACY={best_run['metrics.accuracy']}\n")
        f.write(f"F1_SCORE={best_run['metrics.f1_score']}\n")
        f.write(f"ERROR=Docker build failed\n")
    
    # Create a flag file even if Docker build fails
    with open("/mlflow/training_complete.flag", "w") as f:
        f.write(f"Training completed at: {pd.Timestamp.now()}\n")
        f.write(f"Best model: {model_type}\n")
        f.write(f"Run ID: {run_id}\n") 
        f.write(f"F1 Score: {best_run['metrics.f1_score']}\n")
        f.write(f"Docker build failed with error: {str(e)}\n")
    
    print("Model training completed, but Docker build failed.")
    print("Model information saved to /mlflow/current_model_info.txt")
    print("Training flag created at /mlflow/training_complete.flag")