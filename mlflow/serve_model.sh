#!/bin/bash

# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow --host 0.0.0.0 --port 5000

# Wait for MLflow server to start
sleep 10

# Serve the model (replace with your model URI)
mlflow models serve -m best_model -p 8080 --host 0.0.0.0