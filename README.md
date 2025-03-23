# MLOps Income Prediction Platform

<div align="center">
  <img src="https://img.shields.io/badge/python-3.9-blue.svg" alt="Python 3.9" />
  <img src="https://img.shields.io/badge/docker-compose-2496ED.svg?logo=docker" alt="Docker Compose" />
  <img src="https://img.shields.io/badge/azure-0078D4.svg?logo=microsoft-azure" alt="Azure" />
  <img src="https://img.shields.io/badge/mlflow-0194E2.svg?logo=mlflow" alt="MLflow" />
  <img src="https://img.shields.io/badge/streamlit-FF4B4B.svg?logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/fastapi-009688.svg?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF.svg?logo=github-actions" alt="GitHub Actions" />
</div>

<div align="center">
  <h3>A full-featured MLOps platform for model training, deployment, and monitoring</h3>
</div>

## 🚀 Overview

The MLOps Income Prediction platform demonstrates a complete machine learning operations pipeline, from data processing to model training, deployment, and monitoring. The system uses an income prediction model as a showcase, predicting whether an individual's income exceeds $50K based on census data.

The platform includes:

- Containerized microservices architecture
- Continuous integration and delivery pipelines
- Model training and experimentation
- Model registry and versioning
- Real-time monitoring and metrics visualization
- Automated retraining workflows

## 🏗️ Architecture

![image](https://github.com/user-attachments/assets/6f218666-4c3e-4824-9b3c-058735fc1e55)

The platform consists of the following components, to be deployed locally and on Azure Virtual Machines:

### 📱 Frontend Service
- Streamlit-based user interface
- Interactive form for income predictions

### 🔌 Backend Service
- FastAPI application for prediction serving
- Data validation and transformation
- Prometheus metrics instrumentation

### 📊 MLflow Service
- Experiment tracking and model registry
- Hyperparameter optimization
- Model versioning and artifact storage

### 🤖 Model Service
- Containerized model deployment
- Serving predictions via REST API
- Metrics collection for model performance

### 📈 Monitoring Stack 
- Prometheus for metrics collection
- Grafana for visualization dashboards
- Real-time performance monitoring

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Model Training**: Scikit-learn, MLflow
- **Data Validation**: Great Expectations
- **Deployment**: Docker, Azure
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana
- **Infrastructure**: Azure Cloud

## 📋 Prerequisites

- Docker and Docker Compose
- Azure CLI (only for recreating Azure Deployment)
- GitHub account for CI/CD (will only work until 23.03.2025 due to the expiry of the Azure subscription)
- Python 3.9+

## 🔧 Installation

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/OEDevelopement/MLOPs.git
   cd MLOPs
   ```

2. Create a virtual environment:
   ```bash
   python -m venv mlops-venv
   source mlops-venv/bin/activate  # On Windows: mlops-venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download, unpack and process the dataset
   ```bash
   mkdir mlflow/data/raw
   mkdir mlflow/data/processed
   kaggle datasets download -d wenruliu/adult-income-dataset -p mlflow/data/raw
   python mlflow/unpack_zip.py
   python mlflow/validate_raw_data.py
   python mlflow/process_data.py
   python mlflow/validate_processed_data.py
   ```   

5. Start the development environment:
   ```bash
   docker-compose up
   ```

6. Access the application:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - MLflow UI: http://localhost:5000
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

### Production Deployment

The platform is designed to be deployed to Azure Virutal Machines
by usinf the GitHub Actions workflow for automatic (Merge/Push to TEST ord PROD) or manual deployment.

## 🔄 CI/CD Workflows

### Continuous Integration (CI)

The CI pipeline runs on pull requests to DEV and TEST branches, performing:

- Code quality & Secrutiy checks (flake8, black, mypy, bandit)
- Download, process and validate kaggle dataset
- Starting docker with docker-compose up build
- Unit / integration tests
- On PR to DEV: Auto Merge after successful run

File: ci.yml


### Continuous Deployment (CD)

The CD pipeline automatically deploys to the matching VM:

- TEST environment: On merge to TEST branch
- PROD environment: On merge to PROD branch

File: cd.yml

### Start and Stop VM

This workflow is for starting or stopping a specific VM and can be triggered manually by each team member. This way everyone can use the deployed version without exceeding the budget.

## 🧪 Model Training

The model training includes:

1. Data preprocessing
2. Hyperparameter optimization across multiple algorithms:
   - Random Forest
   - Logistic Regression
   - Gradient Boosting
3. Model evaluation and selection
4. Automated versioning in MLflow

The model training is done by the mlflow_setup.py file.

This Training starts automatic when starting the docker architecture and will take about 5 - 10 minutes. During this process the for the first time, the frontend will provide an error. For the next time, die last model is saved persistentaly and will be used until the the retraining is done and a new model is saved.

The automated retraining is triggered by a GitHub Action Cron Job (Sun, 00:00) wich will run the retraining.yml file. This retraining downloads, processes and saves the newest data and download it on the VM. Afterwards the relating containers are restarted.


## 📊 Monitoring

The monitoring stack provides real-time insights into:

- Service health status indicators
- API call rates and response times
- Active request gauges
- Prediction success rates
- Error tracking visualizations

Access the Grafana dashboard at http://localhost:3000 using:
- Username: admin
- Password: admin

**Note**: Sometimes there is a bug and you need to edit each visual (hover over it and press e).

## 🌲 Project Structure

The project was 

```
mlops-income-prediction/
├── .github/
│   └── workflows/          # CI/CD workflows
│       ├── ci.yml          # Continuous integration pipeline
│       ├── cd.yml          # Continuous deployment pipeline
│       ├── retraining.yml  # Automated model retraining
│       └── manage_vm.yml   # Start or Stop the specific VM
├── backend/                # FastAPI backend service
│   ├── Dockerfile          # Docker Image for the backend
│   ├── main.py             # Backend code (FastAPI)
│   ├── requirements.txt    # Python dependencies for backend
│   └── test_backend.py     # Backend unit tests
├── frontend/               # Streamlit UI
│   ├── Dockerfile          # Docker Image for the frontend
│   ├── app.py              # Streamlit application
│   └── requirements.txt    # Python dependencies for frontend
├── mlflow/                 # MLflow training and experiments
│   ├── Dockerfile          # Docker Image for MLflow
│   ├── mlflow_setup.py     # Training pipeline script
│   ├── model_validation.py # Model validation script
│   ├── param_grid_functions.py # Hyperparameter optimization utils
│   ├── process_data.py     # Data processing script
│   ├── requirements.txt    # Python dependencies for MLflow
│   ├── test_mlflow.py      # MLflow component tests
│   ├── unpack_zip.py       # Utility to extract data
│   ├── validate_raw_data.py # Raw data validation
│   └── validate_processed_data.py # Processed data validation
├── model_service/          # Model serving
│   ├── Dockerfile          # Docker Image for model service
│   ├── model_metrics.py    # Metrics collection for model
│   ├── requirements.txt    # Python dependencies for model service
│   └── wait_for_model.sh   # Initialization script
├── grafana/                # Monitoring dashboards
│   ├── Dockerfile          # Docker Image for Grafana
│   ├── dashboards/         # Predefined dashboards
│   │   └── dashboard.json  # Main monitoring dashboard
│   ├── provisioning/       # Grafana configuration
│   │   ├── dashboards/     # Dashboard provisioning
│   │   │   └── dashboards.yaml
│   │   └── datasources/    # Data source configuration
│   │       └── datasource.yml
│   ├── provisioning/datasource.yml # Prometheus data source
│   └── verify-startup.sh  # Startup verification script
├── prometheus/             # Metrics collection
│   ├── Dockerfile          # Docker Image for Prometheus
│   └── prometheus.yml      # Prometheus configuration
├── docs/                   # Project documentation
│   ├── component-docs.md   # Detailed component documentation
│   ├── deployment-guide.md # Deployment instructions
│   └── development-guide.md # Development guidelines
├── config.json             # Configuration for Azure resources
├── README.md               # Project overview and instructions
├── docker-compose.yml      # Local development setup
└── Solution_Architecture.drawio # Architecture diagram source
```
## 📄 GitHub Environment Variables

| Variable/Secret | Description | Usage |
|-----------------|-------------|-------|
| `KAGGLE_USERNAME` | Kaggle username for API access | Used to download datasets from Kaggle |
| `KAGGLE_TOKEN` | Authentication token for Kaggle API | Used together with username for Kaggle authentication |
| `KAGGLE_KEY` | Alternative name for Kaggle API token | Used in CI workflow |
| `AZURE_CREDENTIALS` | Azure service principal credentials | Used for authenticating to Azure |
| `VM_SSH_KEY` | SSH private key for VM access | Used for secure SSH access to Azure VMs |
| `VM_HOST` | Hostname or IP address of Azure VM | Target for SSH connections |
| `VM_USER` | Username for SSH login to VM | Used with SSH key for VM access |
| `VM_PROJECT_PATH` | Path to project directory on VM | Specifies where code is deployed on VM |
| `RESOURCE_GROUP` | Azure resource group name | Groups related Azure resources |
| `LOCATION` | Azure region for resources | Determines where resources are deployed |
| `VM_NAME` | Name of the Azure VM | Identifies which VM to start/stop |
| `STORAGE_ACCOUNT` | Azure Storage account name | Used for storing model data |
| `CONTAINER` | Blob container for processed data | Stores processed training data |
| `CONTAINER_RAW` | Blob container for raw data | Stores raw training data |

These variables are used across the CI/CD workflows for deployment, model retraining, and VM management. They're configured as repository secrets (for sensitive information) and environment variables (for non-sensitive configuration) in the GitHub repository settings.

## 📄 Environment Variables (Docker)

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | URI for MLflow tracking server | `http://mlflow:5000` |
| `MODEL_PATH` | Path to the deployed model | `/app/models/best_model` |
| `BACKEND_URL` | URL for the backend service | `http://backend:8000` |
| `API_URL` | URL for the prediction API | `http://backend:8000/predict` |

## 👥 Project Contributors

| Contributor | Areas of Responsibility |
|-------------|--------------------------|
| Louis Baars | Frontend & Lead for documentation |
| Sophie Bayersdörfer | GitHub Actions & Azure Deployment |
| Joel Rasch | Backend & Monitoring |
| John Titz | Docker Development|
| Yanoothan Yarlvarathan | Data Processing & Model Training |

This project was developed as part of the course Machine Learning Operations at the FH SWF. For questions about specific components, please reach out to the responsible contributor listed above.

## 🙏 Acknowledgements

- [MLflow](https://mlflow.org/) for experiment tracking
- [Streamlit](https://streamlit.io/) for the frontend interface
- [FastAPI](https://fastapi.tiangolo.com/) for the backend API
- [Docker](https://www.docker.com/) for containerization
- [Azure](https://azure.microsoft.com/) for cloud infrastructure
- [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/) for monitoring
