#!/bin/bash
set -e  # Exit on any error

# Configuration variables - can be overridden with environment variables
RESOURCE_GROUP=${RESOURCE_GROUP:-"mlops-rg-dev"}
ENVIRONMENT_NAME=${ENVIRONMENT_NAME:-"mlops-env-dev"}
ACR_NAME=${ACR_NAME:-"mlopscrafter25acrdev"}
ACR_LOGIN_SERVER=${ACR_LOGIN_SERVER:-"mlopscrafter25acrdev.azurecr.io"}
LOCATION=${LOCATION:-"westeurope"}
TAG=${TAG:-"latest"}

echo "=== MLOps Platform Deployment ==="
echo "Resource Group: $RESOURCE_GROUP"
echo "Environment: $ENVIRONMENT_NAME"
echo "ACR: $ACR_NAME"
echo "Location: $LOCATION"

# Ensure Azure CLI extensions are installed
echo "Installing required Azure CLI extensions..."
az extension add --name containerapp --upgrade --yes

# Create or update resource group
echo "Checking resource group..."
if ! az group show --name $RESOURCE_GROUP &>/dev/null; then
  echo "Creating resource group $RESOURCE_GROUP in $LOCATION"
  az group create --name $RESOURCE_GROUP --location $LOCATION
else
  echo "Resource group $RESOURCE_GROUP already exists"
fi

# Set up ACR
echo "Setting up Azure Container Registry..."
if ! az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP &>/dev/null; then
  echo "Creating ACR $ACR_NAME"
  ACR_LOGIN_SERVER=$(az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --query loginServer --output tsv)
else
  echo "ACR $ACR_NAME already exists"
  ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query loginServer --output tsv)
fi

# Enable admin access to ACR
echo "Enabling admin access to ACR..."
az acr update --name $ACR_NAME --admin-enabled true

# Get ACR credentials
echo "Getting ACR credentials..."
ACR_USERNAME=$ACR_NAME
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query 'passwords[0].value' -o tsv)

# Login to ACR
echo "Logging in to ACR..."
echo $ACR_PASSWORD | docker login $ACR_LOGIN_SERVER --username $ACR_USERNAME --password-stdin

# Build and push Docker images
echo "Building and pushing Docker images..."
services=("frontend" "backend" "mlflow" "model_service" "prometheus" "grafana")

for service in "${services[@]}"; do
  echo "Building and pushing $service..."
  
  # Service directory might be named differently in some cases
  if [ "$service" = "model_service" ]; then
    service_dir="./$service"
  else
    service_dir="./$service"
  fi
  
  # Build the Docker image
  docker build -t $ACR_LOGIN_SERVER/$service:$TAG $service_dir
  
  # Push the image to ACR
  docker push $ACR_LOGIN_SERVER/$service:$TAG
done

# Use existing Container Apps Environment
echo "Using existing Container Apps Environment: $ENVIRONMENT_NAME..."
if ! az containerapp env show --name $ENVIRONMENT_NAME --resource-group $RESOURCE_GROUP &>/dev/null; then
  echo "ERROR: Container Apps Environment $ENVIRONMENT_NAME does not exist in resource group $RESOURCE_GROUP"
  echo "Please create it manually or use an existing environment"
  exit 1
else
  echo "Container Apps Environment $ENVIRONMENT_NAME exists, proceeding with deployment"
fi

# Deploy each service as a Container App
echo "Deploying services as Container Apps..."

# Deploy MLflow first (as it's a dependency)
echo "Deploying MLflow service..."
az containerapp create \
  --name mlflow \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT_NAME \
  --image $ACR_LOGIN_SERVER/mlflow:$TAG \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 5000 \
  --ingress external \
  --env-vars "MLFLOW_TRACKING_URI=http://mlflow:5000" \
      "MLFLOW_BACKEND_STORE_URI=/mlflow/mlruns" \
      "MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts" \
      "MODEL_OUTPUT_PATH=/app/models/best_model" \
  --min-replicas 1 \
  --max-replicas 1

MLFLOW_URL=$(az containerapp show --name mlflow --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)
MLFLOW_URL="https://$MLFLOW_URL"
echo "MLflow URL: $MLFLOW_URL"

# Deploy model_service
echo "Deploying model service..."
az containerapp create \
  --name model-service \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT_NAME \
  --image $ACR_LOGIN_SERVER/model_service:$TAG \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8080 \
  --ingress external \
  --env-vars "MODEL_PATH=/app/models/best_model" \
      "MLFLOW_TRACKING_URI=$MLFLOW_URL" \
  --min-replicas 1 \
  --max-replicas 1

MODEL_URL=$(az containerapp show --name model-service --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)
MODEL_URL="https://$MODEL_URL"
echo "Model Service URL: $MODEL_URL"

# Deploy backend
echo "Deploying backend service..."
az containerapp create \
  --name backend \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT_NAME \
  --image $ACR_LOGIN_SERVER/backend:$TAG \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8000 \
  --ingress external \
  --env-vars "MODEL_SERVICE_URL=$MODEL_URL/invocations" \
  --min-replicas 1 \
  --max-replicas 3

BACKEND_URL=$(az containerapp show --name backend --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)
BACKEND_URL="https://$BACKEND_URL"
echo "Backend URL: $BACKEND_URL"

# Deploy prometheus
echo "Deploying Prometheus..."
az containerapp create \
  --name prometheus \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT_NAME \
  --image $ACR_LOGIN_SERVER/prometheus:$TAG \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 9090 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 1

PROMETHEUS_URL=$(az containerapp show --name prometheus --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)
PROMETHEUS_URL="https://$PROMETHEUS_URL"
echo "Prometheus URL: $PROMETHEUS_URL"

# Deploy grafana
echo "Deploying Grafana..."
az containerapp create \
  --name grafana \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT_NAME \
  --image $ACR_LOGIN_SERVER/grafana:$TAG \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 3000 \
  --ingress external \
  --env-vars "GF_SECURITY_ADMIN_USER=admin" \
      "GF_SECURITY_ADMIN_PASSWORD=admin" \
      "GF_USERS_ALLOW_SIGN_UP=false" \
  --min-replicas 1 \
  --max-replicas 1

GRAFANA_URL=$(az containerapp show --name grafana --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)
GRAFANA_URL="https://$GRAFANA_URL"
echo "Grafana URL: $GRAFANA_URL"

# Finally, deploy frontend
echo "Deploying frontend service..."
az containerapp create \
  --name frontend \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT_NAME \
  --image $ACR_LOGIN_SERVER/frontend:$TAG \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8501 \
  --ingress external \
  --env-vars "BACKEND_URL=$BACKEND_URL" \
      "API_URL=$BACKEND_URL/predict" \
  --min-replicas 1 \
  --max-replicas 3

FRONTEND_URL=$(az containerapp show --name frontend --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)
FRONTEND_URL="https://$FRONTEND_URL"
echo "Frontend URL: $FRONTEND_URL"

# Verify deployment
echo "=== Deployment Summary ==="
echo "Frontend: $FRONTEND_URL"
echo "Backend: $BACKEND_URL"
echo "MLflow: $MLFLOW_URL"
echo "Model Service: $MODEL_URL"
echo "Prometheus: $PROMETHEUS_URL"
echo "Grafana: $GRAFANA_URL (admin/admin)"

echo "Testing frontend availability..."
curl -IL $FRONTEND_URL

echo "Deployment completed!"