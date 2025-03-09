FROM python:3.9-slim

WORKDIR /app

# Systemabhängigkeiten installieren (curl hinzugefügt!)
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Docker-CLI installieren
RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
RUN echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt-get update && apt-get install -y docker-ce-cli && rm -rf /var/lib/apt/lists/*

# Installiere Python-Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere Skripte
COPY mlflow_setup.py param_grid_functions.py ./

# Erstelle Verzeichnisse für MLflow
RUN mkdir -p /mlflow /app/data /mlflow/model_dockerfile

# Port freigeben
EXPOSE 5000

# Starte MLflow-Server und dann das Training
CMD mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}" \
    --host 0.0.0.0 \
    --port 5000 & \
    sleep 15 && \
    if [ -f "/app/data/processed/processed_data.csv" ]; then \
        python mlflow_setup.py; \
    else \
        echo "Datendatei nicht gefunden"; \
        find /app/data -type f; \
    fi && \
    tail -f /dev/null