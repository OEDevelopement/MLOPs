FROM python:3.9-slim

WORKDIR /app

# Minimale Abhängigkeiten installieren
RUN pip install --no-cache-dir flask requests

# Kopieren des Placeholder-Services
COPY placeholder_service.py .

# Port freigeben
EXPOSE 8080

# Placeholder-Service starten
CMD ["python", "placeholder_service.py"]