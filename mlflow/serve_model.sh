#!/bin/bash

# Start MLflow server
mlflow server \
    --host 0.0.0.0 \
    --port 5001 \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT} &

# Wait for MLflow server to start
sleep 5

python mlflow_setup.py
python model_validation.py

sleep 10

# Serve the model (replace with your model URI)
mlflow models serve -m best_model -p 8080 --host 0.0.0.0 --no-conda