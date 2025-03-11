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
- A/B testing capabilities
- Real-time monitoring and metrics visualization
- Automated retraining workflows

![Architecture Diagram](docs/architecture_diagram.png)

## 🏗️ Architecture

The platform consists of the following components:

### 📱 Frontend Service
- Streamlit-based user interface
- Interactive form for income predictions
- Real-time system monitoring dashboard

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

### 📈 Monitoring Stack (LGTM)
- Prometheus for metrics collection
- Grafana for visualization dashboards
- Real-time performance monitoring

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Model Training**: Scikit-learn, MLflow
- **Deployment**: Docker, Azure Container Apps
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana
- **Infrastructure**: Azure Cloud

## 📋 Prerequisites

- Docker and Docker Compose
- Azure CLI
- GitHub account for CI/CD
- Python 3.9+

## 🔧 Installation

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mlops-income-prediction.git
   cd mlops-income-prediction
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

4. Start the development environment:
   ```bash
   docker-compose up
   ```

5. Access the application:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - MLflow UI: http://localhost:5000
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

### Production Deployment

The platform is designed to be deployed to Azure Container Apps:

1. Configure your Azure credentials:
   ```bash
   az login
   ```

2. Use the deployment script:
   ```bash
   ./DeployContainerAppDEV  # For DEV environment
   ```

Or use the GitHub Actions workflow for automatic deployment.

## 🔄 CI/CD Workflows

### Continuous Integration (CI)

The CI pipeline runs on pull requests to DEV and TEST branches, performing:

- Code quality checks (flake8, black, mypy, bandit)
- Unit and integration tests
- Security scanning
- Docker image builds

### Continuous Deployment (CD)

The CD pipeline automatically deploys to:

- DEV environment: On merge to DEV branch
- TEST environment: On merge to TEST branch
- PROD environment: On merge to PROD branch

## 🧪 Model Training

The model training pipeline includes:

1. Data preprocessing
2. Hyperparameter optimization across multiple algorithms:
   - Random Forest
   - Logistic Regression
   - Gradient Boosting
3. Model evaluation and selection
4. Automated versioning in MLflow

To manually trigger model retraining:

```bash
# Using GitHub Actions
gh workflow run retraining.yml

# Or locally
docker-compose exec mlflow python mlflow_setup.py
```

## 📊 Monitoring

The monitoring stack provides real-time insights into:

- System health and performance
- Model accuracy and drift
- Prediction latency
- User engagement metrics

Access the Grafana dashboard at http://localhost:3000 using:
- Username: admin
- Password: admin

## 🌲 Project Structure

```
mlops-income-prediction/
├── .github/
│   └── workflows/          # CI/CD workflows
│       ├── ci.yml          # Continuous integration pipeline
│       ├── cd.yml          # Continuous deployment pipeline
│       └── retraining.yml  # Automated model retraining
├── backend/                # FastAPI backend service
├── frontend/               # Streamlit UI
├── mlflow/                 # MLflow training and experiments
├── model_service/          # Model serving
├── grafana/                # Monitoring dashboards
├── prometheus/             # Metrics collection
├── tests/                  # Test suite
└── docker-compose.yml      # Local development setup
```

## 📄 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | URI for MLflow tracking server | `http://mlflow:5000` |
| `MODEL_PATH` | Path to the deployed model | `/app/models/best_model` |
| `BACKEND_URL` | URL for the backend service | `http://backend:8000` |
| `API_URL` | URL for the prediction API | `http://backend:8000/predict` |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Documentation

For more detailed documentation, see the `docs/` directory:

- [Architecture Overview](docs/architecture.md)
- [Development Guide](docs/development.md)
- [Deployment Guide](docs/deployment.md)
- [Model Training Guide](docs/model_training.md)
- [Monitoring Guide](docs/monitoring.md)

## 🙏 Acknowledgements

- [MLflow](https://mlflow.org/) for experiment tracking
- [Streamlit](https://streamlit.io/) for the frontend interface
- [FastAPI](https://fastapi.tiangolo.com/) for the backend API
- [Docker](https://www.docker.com/) for containerization
- [Azure](https://azure.microsoft.com/) for cloud infrastructure
- [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/) for monitoring