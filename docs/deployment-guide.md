# Deployment Guide

This guide covers the deployment process for the MLOps Income Prediction platform, including infrastructure setup, environment configuration, and deployment workflows.

## Table of Contents

1. [Deployment Environments](#deployment-environments)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Deployment Methods](#deployment-methods)
   - [Manual Deployment](#manual-deployment)
   - [GitHub Actions Deployment](#github-actions-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Monitoring During Deployment](#monitoring-during-deployment)
6. [Rollback Procedures](#rollback-procedures)
7. [Troubleshooting](#troubleshooting)

## Deployment Environments

The platform supports three deployment environments:

| Environment | Branch | Purpose | Resource Location |
|-------------|--------|---------|-----------------|
| DEV | DEV | Development and testing | westus2 |
| TEST | TEST | QA and staging | westeurope |
| PROD | PROD | Production | northeurope |

## Infrastructure Setup

### Pre-requisites

1. **Azure Subscription** with appropriate permissions
2. **Azure CLI** installed and configured
3. **GitHub** repository connected to Azure (for GitHub Actions)

### Required Azure Resources

- **Resource Group:** Separate for each environment
- **Azure Container Registry (ACR):** For storing Docker images
- **Azure Container Apps (ACA):** For running containerized services
- **Persistent Storage:** For MLflow artifacts and model storage

## Deployment Methods

### Manual Deployment

Use the provided shell script for manual deployment:

```bash
# For DEV environment
./DeployContainerAppDEV

# For TEST environment (modify script as needed)
POSTFIX=test ./DeployContainerAppDEV

# For PROD environment (modify script as needed)
POSTFIX=prod ./DeployContainerAppDEV
```

The script performs the following actions:
1. Creates or updates the resource group
2. Creates or gets the Azure Container Registry
3. Builds and pushes Docker images
4. Deploys to Azure Container Apps using docker-compose

### GitHub Actions Deployment

The CD pipeline (`cd.yml`) handles automated deployments:

1. **Trigger:**
   - Push to TEST or PROD branch
   - Manual trigger via workflow_dispatch

2. **Environment Determination:**
   - Based on branch name or manual input

3. **Build and Deploy:**
   - Checks out code
   - Sets up Docker and Azure CLI
   - Determines environment-specific variables (postfix, location)
   - Creates/updates resource group
   - Creates/updates ACR and pushes images
   - Deploys to Azure Container Apps
   - Returns deployment URL

## Environment Configuration

Environment-specific configuration is managed through:

1. **Environment Variables:** Set in `docker-compose-{env}.yml` files
2. **Docker Compose Overrides:** Different compose files per environment
3. **GitHub Actions Variables:** Set based on environment determination

Key configuration differences between environments:

| Setting | DEV | TEST | PROD |
|---------|-----|------|------|
| Resource Group | mlops-rg-dev | mlops-rg-test | mlops-rg-prod |
| ACR Name | mlops25acrdev | mlops25acrtest | mlops25acrprod |
| Location | westus2 | westeurope | northeurope |
| Scale | Minimal | Medium | High Availability |

## Monitoring During Deployment

During deployment, monitor:

1. **GitHub Actions Console:** For build and deployment logs
2. **Azure Portal:** For resource creation and status
3. **Application Logs:** Via Azure Container Apps logs
4. **Health Endpoints:** Check `/health` endpoints after deployment

## Rollback Procedures

If deployment fails or issues are found:

### Using GitHub Actions:

1. Trigger the CD workflow with a previous commit:
   ```bash
   gh workflow run cd.yml -f ref=<previous-commit-sha>
   ```

### Manual Rollback:

1. Return to the previous Docker image tags:
   ```bash
   az containerapp update --name <app-name> --resource-group <resource-group> --image <acr-name>.azurecr.io/<service>:<previous-tag>
   ```

## Troubleshooting

### Common Issues

1. **Image Push Failures:**
   - Check ACR credentials and permissions
   - Verify Docker daemon is running
   - Check image build logs

2. **Deployment Failures:**
   - Examine Azure CLI errors
   - Check resource quotas/limits
   - Verify service dependencies

3. **Application Startup Failures:**
   - Check container logs in Azure Portal
   - Verify environment variables
   - Test service health endpoints

### Debugging Tips

1. **Test Locally First:**
   ```bash
   docker-compose -f docker-compose-{env}.yml up
   ```

2. **Check Azure Container App Logs:**
   ```bash
   az containerapp logs show --name <app-name> --resource-group <resource-group> --follow
   ```

3. **Verify Network Connectivity:**
   ```bash
   az containerapp exec --name <app-name> --resource-group <resource-group> --command sh
   # Then use curl to test internal endpoints
   ```

4. **Check Resource Configuration:**
   ```bash
   az containerapp show --name <app-name> --resource-group <resource-group>
   ```
