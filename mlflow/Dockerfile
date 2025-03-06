FROM python:3.9-slim

WORKDIR /mlflow_app

# MLflow installieren
RUN pip install --no-cache-dir mlflow scikit-learn

# Falls du noch weitere Packages brauchst:
# RUN pip install <deine-weitere-packages>

COPY serve_model.sh .
RUN chmod +x serve_model.sh

# Exponiere Port für MLflow
EXPOSE 5000
EXPOSE 8080

# Standard-Kommando: MLflow-Server starten
CMD ["./serve_model.sh"]
