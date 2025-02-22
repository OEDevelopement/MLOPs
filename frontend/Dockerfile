# Beispiel: Python-Image
FROM python:3.9-slim

# Arbeitsverzeichnis
WORKDIR /app

# Anforderungen installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Quellcode kopieren
COPY . .

# Standard-Port für Streamlit
EXPOSE 8501

# Startbefehl
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
