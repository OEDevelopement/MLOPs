#!/bin/bash

# Modell-Pfad aus Umgebungsvariable oder Standard-Pfad
MODEL_PATH=${MODEL_PATH:-/app/models/best_model}

echo "Starte Model Service Prüfung..."
echo "Warte auf Modell in: $MODEL_PATH"

# Anzahl der Versuche
MAX_ATTEMPTS=100
ATTEMPT=0

# Verzeichnis erstellen, falls es noch nicht existiert
mkdir -p "$MODEL_PATH"

# In Schleife prüfen, ob das Modell existiert
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT+1))
    
    echo "Versuch $ATTEMPT von $MAX_ATTEMPTS..."
    
    if [ -f "$MODEL_PATH/MLmodel" ]; then
        echo "✅ MODELL GEFUNDEN! STARTE SERVING..."
        echo "Modell-Dateien:"
        ls -la "$MODEL_PATH"
        break
    else
        echo "❌ Warte auf Modell-Dateien... Schlafe 5 Sekunden"
        sleep 5
    fi
done

if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
    echo "❌ Maximale Anzahl von Versuchen erreicht. Modell konnte nicht gefunden werden."
    exit 1
fi

# Modell Serving starten
echo "Starte MLflow Model Serving für $MODEL_PATH..."
exec mlflow models serve -m "$MODEL_PATH" -h 0.0.0.0 -p 8080 --no-conda
