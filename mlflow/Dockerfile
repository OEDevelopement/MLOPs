FROM python:3.9-slim

WORKDIR /mlflow_app

# MLflow installieren
RUN pip install --no-cache-dir mlflow

# Falls du noch weitere Packages brauchst:
# RUN pip install <deine-weitere-packages>

# Exponiere Port für MLflow
EXPOSE 5000

# Standard-Kommando: MLflow-Server starten
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlflow", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
