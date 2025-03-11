FROM python:3.9-slim

WORKDIR /app

# Benötigte Pakete
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skript kopieren
COPY wait_for_model.sh .
RUN chmod +x wait_for_model.sh

# Modell-Pfad setzen
ENV MODEL_PATH=/app/models/best_model

# Skript starten
CMD ["./wait_for_model.sh"]