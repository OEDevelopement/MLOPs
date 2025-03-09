from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({
        "status": "placeholder",
        "message": "Placeholder model service running. Awaiting trained model."
    })

@app.route('/invocations', methods=['POST'])
def invocations():
    # Einfache Dummy-Antwort, die das gleiche Format wie das echte Modell hat
    request_data = request.get_json(silent=True)
    
    if request_data and 'dataframe_split' in request_data:
        # Anzahl der Datenpunkte ermitteln
        data = request_data['dataframe_split'].get('data', [[]])
        num_predictions = len(data)
        
        # Dummy-Vorhersagen (alle 0)
        predictions = [0] * num_predictions
    else:
        predictions = [0]  # Standardantwort
    
    return jsonify({
        "predictions": predictions, 
        "message": "This is a placeholder. Actual model not yet trained.",
        "is_placeholder": True
    })

if __name__ == '__main__':
    print("Starting placeholder model service...")
    app.run(host='0.0.0.0', port=8080)