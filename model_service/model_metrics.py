import os
import json
import datetime
from prometheus_client import Counter, Gauge, start_http_server

# Model deployment tracking with simple metrics
MODEL_DEPLOYMENTS = Counter('model_deployments_total', 'Total number of model deployments')
MODEL_CHANGED = Counter('model_changed_total', 'Number of times the model was changed')
CURRENT_MODEL_AGE = Gauge('current_model_age_seconds', 'Age of current model in seconds')

# File to store model metadata
MODEL_INFO_FILE = "/app/models/model_info.json"

def track_model_deployment():
    """
    Track when a new model is deployed by checking for changes
    """
    model_path = os.environ.get("MODEL_PATH", "/app/models/best_model")
    
    # Default model info
    model_info = {
        "last_modified": None,
        "deployment_time": None,
        "deployment_count": 0
    }
    
    # Try to load existing model info
    if os.path.exists(MODEL_INFO_FILE):
        try:
            with open(MODEL_INFO_FILE, 'r') as f:
                model_info = json.load(f)
        except:
            pass
    
    # Check if model exists
    if os.path.exists(model_path):
        # Get the latest modification time of any file in the model directory
        latest_mod_time = max(
            os.path.getmtime(os.path.join(dirpath, f))
            for dirpath, dirnames, filenames in os.walk(model_path)
            for f in filenames
        )
        
        # Format the timestamp
        latest_mod_time_str = datetime.datetime.fromtimestamp(latest_mod_time).isoformat()
        
        # If model has changed or is new
        if model_info["last_modified"] != latest_mod_time_str:
            print(f"Detected new or changed model at {latest_mod_time_str}")
            
            # Update model info
            model_info["last_modified"] = latest_mod_time_str
            model_info["deployment_time"] = datetime.datetime.now().isoformat()
            model_info["deployment_count"] += 1
            
            # Increment Prometheus counters
            MODEL_DEPLOYMENTS.inc()
            MODEL_CHANGED.inc()
            
            # Save updated info
            os.makedirs(os.path.dirname(MODEL_INFO_FILE), exist_ok=True)
            with open(MODEL_INFO_FILE, 'w') as f:
                json.dump(model_info, f)
        
        # Update model age metric
        if model_info["deployment_time"]:
            deploy_time = datetime.datetime.fromisoformat(model_info["deployment_time"])
            age_seconds = (datetime.datetime.now() - deploy_time).total_seconds()
            CURRENT_MODEL_AGE.set(age_seconds)

# Start metrics server and tracking when imported
def start_metrics_server(port=9000):
    """Start a Prometheus metrics endpoint"""
    try:
        print(f"Starting metrics server on port {port}")
        start_http_server(port)
        
        # Track initial model state
        track_model_deployment()
        
        print("Model deployment tracking started")
        return True
    except Exception as e:
        print(f"Failed to start metrics server: {e}")
        return False

if __name__ == "__main__":
    start_metrics_server()
    
    # Keep checking for model changes periodically
    import time
    while True:
        track_model_deployment()
        time.sleep(60)  # Check every minute