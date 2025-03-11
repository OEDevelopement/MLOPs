# Component Documentation

This document provides detailed information about each component in the MLOps Income Prediction platform.

## Table of Contents

1. [GitHub Workflows](#github-workflows)
2. [Frontend Service](#frontend-service)
3. [Backend Service](#backend-service)
4. [MLflow Service](#mlflow-service)
5. [Model Service](#model-service)
6. [Monitoring Stack](#monitoring-stack)
7. [Testing Framework](#testing-framework)

## GitHub Workflows

### CI Pipeline (`ci.yml`)

The Continuous Integration pipeline automatically runs on pull requests to DEV and TEST branches, and can also be triggered manually.

**Key Features:**
- **Code Quality Checks:** Runs flake8, black, mypy, and bandit for code quality and security scanning.
- **Integration Tests:** Executes tests for each component to ensure compatibility.
- **Auto-Merge:** Automatically merges PRs to the DEV branch when all tests pass.

**Notable Code Sections:**
```yaml
code-quality:
  name: Code Quality Checks
  # ... checks for syntax, style, formatting, types, and security

integration-tests:
  name: Integration Tests
  needs: code-quality
  # ... runs component tests with Docker Compose

auto-merge:
  name: Auto-Merge to DEV
  if: github.base_ref == 'DEV' && success()
  # ... automatically merges passing PRs
```

### CD Pipeline (`cd.yml`)

The Continuous Deployment pipeline deploys the application to Azure Container Apps when code is pushed to TEST or PROD branches.

**Key Features:**
- **Environment Detection:** Automatically determines deployment environment based on branch or manual input.
- **Docker Build & Push:** Builds container images and pushes them to Azure Container Registry.
- **Azure Deployment:** Deploys to Azure Container Apps with environment-specific configurations.

**Notable Code Sections:**
```yaml
determine_env:
  # Determines environment (TEST/PROD) from branch or manual input

resource_names:
  # Sets resource names and locations based on environment
  
build-and-push:
  # Builds and pushes Docker images for each service

deploy:
  # Creates or updates container apps environment using docker-compose
```

### Retraining Workflow (`retraining.yml`)

Automatically retrains the ML model on a schedule or manual trigger.

**Key Features:**
- **Model Training:** Executes the MLflow training script to find the best model.
- **Validation:** Validates model performance before deployment.
- **Automated Deployment:** Deploys the new model if it performs better than the previous one.

## Frontend Service

A Streamlit-based user interface for income prediction and system monitoring.

**Key Features:**
- **Prediction Form:** Interactive form for users to input data for income prediction.
- **Monitoring Dashboard:** Real-time system metrics and model performance visualization.
- **Prometheus Integration:** Exposes metrics for frontend usage.

**Notable Code Sections:**
```python
# Metrics setup for Prometheus monitoring
PAGE_VIEWS = Counter('frontend_page_views_total', 'Total number of page views', registry=custom_registry)
PREDICTIONS = Counter('frontend_predictions_total', 'Total number of predictions made', registry=custom_registry)

# API endpoint call
response = requests.post(API_URL, json=data)
result = response.json()
prediction = result.get("predictions", [0])[0]

# Monitoring page
def create_monitoring_page():
    # ... retrieves and displays backend metrics, prediction success rates, etc.
```

## Backend Service

FastAPI application that handles prediction requests and communicates with the model service.

**Key Features:**
- **REST API:** Provides endpoints for predictions and system health.
- **Data Validation:** Uses Pydantic models to validate incoming data.
- **Prometheus Metrics:** Tracks API calls, active requests, and prediction errors.

**Notable Code Sections:**
```python
# Metrics for monitoring
API_CALLS = Counter('api_calls_total', 'Total number of API calls', ['endpoint'])
PREDICTIONS = Counter('predictions_total', 'Total number of predictions made')
PREDICTION_ERRORS = Counter('prediction_errors_total', 'Total number of prediction errors')

# Prediction endpoint
@app.post("/predict")
async def predict(input_person: Person):
    # ... processes input data, sends to model service, and returns prediction
    
# Metrics middleware for request tracking
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    # ... tracks request metrics
```

## MLflow Service

Handles model training, experiment tracking, and model registry.

**Key Features:**
- **Experiment Tracking:** Logs model parameters, metrics, and artifacts.
- **Hyperparameter Optimization:** Tests multiple models with various configurations.
- **Model Registry:** Stores and versions trained models.
- **Model Validation:** Evaluates model performance with cross-validation.

**Notable Code Sections:**
```python
# Model configurations for hyperparameter optimization
model_configs = [
    ("random_forest", RandomForestClassifier, param_combinations_rf, param_grid_rf),
    ("logistic_regression", LogisticRegression, param_combinations_lr, param_grid_lr),
    ("gradient_boosting", GradientBoostingClassifier, param_combinations_gb, param_grid_gb)
]

# MLflow runs for each model configuration
for model_type, model_class, param_combinations, param_grid in model_configs:
    for params in param_combinations:
        with mlflow.start_run():
            # ... trains model, logs metrics, and saves the model

# Best model selection and saving
best_run = runs.loc[runs["metrics.f1_score"].idxmax()]
model_uri = f"runs:/{run_id}/{model_type}_pipeline"
mlflow.sklearn.save_model(
    sk_model=mlflow.sklearn.load_model(model_uri),
    path=model_output_path
)
```

## Model Service

Serves the trained model for predictions via a REST API.

**Key Features:**
- **Model Loading:** Loads the best model from the MLflow registry.
- **Prediction Serving:** Provides an endpoint for making predictions.
- **Metrics Collection:** Tracks model usage and performance.
- **Automatic Updates:** Detects and loads new models as they become available.

**Notable Code Sections:**
```python
# Model deployment tracking
def track_model_deployment():
    # ... checks for model changes and updates metrics
    
    # If model has changed
    if model_info["last_modified"] != latest_mod_time_str:
        # ... updates model info and increments counters
        MODEL_DEPLOYMENTS.inc()
        MODEL_CHANGED.inc()

# wait_for_model.sh script
# ... waits for model files and starts MLflow model serving
exec mlflow models serve -m "$MODEL_PATH" -h 0.0.0.0 -p 8080 --no-conda
```

## Monitoring Stack

Collects and visualizes metrics from all system components.

### Prometheus

Scrapes metrics from services and stores time-series data.

**Key Features:**
- **Metric Collection:** Gathers metrics from all services.
- **Data Storage:** Stores time-series data for visualization.
- **Alert Rules:** Defines thresholds for alerting (configured in `prometheus.yml`).

### Grafana

Visualizes metrics from Prometheus in interactive dashboards.

**Key Features:**
- **Dashboards:** Pre-configured dashboards for system and model monitoring.
- **Alerts:** Visual indicators for system or model issues.
- **User Authentication:** Secure access to monitoring data.

**Notable Dashboard Panels:**
- Frontend Usage Rate
- Services Status
- Model Performance Metrics
- API Call Rates

## Testing Framework

Comprehensive test suite for all components.

**Key Features:**
- **Unit Tests:** Tests individual functions and classes.
- **Integration Tests:** Tests interactions between components.
- **CI Integration:** Automatically runs tests on pull requests.

**Notable Test Cases:**
```python
# Backend API tests
def test_predict_endpoint_success(mock_post):
    # ... tests successful prediction flow

def test_predict_endpoint_mlflow_error(mock_post):
    # ... tests error handling

# MLflow parameter validation tests
def test_is_valid_lr_params():
    # ... tests parameter validation logic
```
