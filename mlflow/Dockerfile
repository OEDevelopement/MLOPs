FROM python:3.9-slim

WORKDIR /app

# Nur wirklich notwendige Systemabhängigkeiten
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Installiere Python-Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere Skripte
COPY . .

# Port freigeben
EXPOSE 5000

# Starte MLflow-Server und dann das Training
CMD mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}" \
    --host 0.0.0.0 \
    --port 5000 & \
    sleep 15 && \
    python validate_raw_data.py && \
    python process_data.py && \
    python validate_processed_data.py && \
    python mlflow_setup.py && \
    python model_validation.py && \
    tail -f /dev/null
