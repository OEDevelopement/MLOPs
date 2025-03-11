# Development Guide

This guide provides instructions and best practices for developing and extending the MLOps Income Prediction platform.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Docker Development](#docker-development)
7. [Adding New Features](#adding-new-features)
8. [Troubleshooting](#troubleshooting)

## Development Environment Setup

### Prerequisites

- **Python 3.9+**
- **Docker** and **Docker Compose**
- **Git**
- **Visual Studio Code** (recommended) or your preferred IDE
- **Azure CLI** (for deployment testing)

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mlops-income-prediction.git
   cd mlops-income-prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv mlops-venv
   source mlops-venv/bin/activate  # On Windows: mlops-venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r dev-requirements.txt  # Development tools
   ```

4. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

5. **Configure environment variables:**
   Create a `.env` file in the project root with:
   ```
   MLFLOW_TRACKING_URI=http://localhost:5000
   MODEL_PATH=./models/best_model
   ```

## Project Structure

The project follows a microservices architecture with these key components:

```
mlops-income-prediction/
├── .github/workflows/        # CI/CD pipelines
├── backend/                  # FastAPI backend service
│   ├── Dockerfile            # Container definition
│   ├── main.py               # API implementation
│   └── requirements.txt      # Python dependencies
├── frontend/                 # Streamlit UI
│   ├── Dockerfile
│   ├── app.py                # Streamlit application
│   └── requirements.txt
├── mlflow/                   # MLflow training service
│   ├── Dockerfile
│   ├── mlflow_setup.py       # Training pipeline
│   ├── param_grid_functions.py  # Hyperparameter utils
│   └── requirements.txt
├── model_service/            # Model serving
│   ├── Dockerfile
│   ├── model_metrics.py      # Metrics collection
│   ├── wait_for_model.sh     # Service initialization
│   └── requirements.txt
├── monitoring/               # Prometheus & Grafana
│   ├── grafana/              # Dashboards & config
│   └── prometheus/           # Metrics collection
├── tests/                    # Test suite
│   ├── test_backend.py
│   └── test_mlflow.py
└── docker-compose.yml        # Service definitions
```

## Development Workflow

### Git Branching Strategy

We use a branch-based workflow:

1. **Main Branches:**
   - `DEV`: Development branch, main integration point
   - `TEST`: Testing/staging environment
   - `PROD`: Production environment

2. **Feature Branches:**
   - Create from `DEV`: `feature/feature-name`
   - Pull request back to `DEV`

3. **Bugfix Branches:**
   - Create from affected environment: `bugfix/bug-description`
   - Pull request to affected environment

### Development Cycle

1. **Create a feature branch:**
   ```bash
   git checkout DEV
   git pull
   git checkout -b feature/your-feature-name
   ```

2. **Implement your changes**

3. **Run tests locally:**
   ```bash
   pytest
   ```

4. **Commit changes with meaningful messages:**
   ```bash
   git add .
   git commit -m "Add feature X to solve problem Y"
   ```

5. **Push branch and create PR:**
   ```bash
   git push -u origin feature/your-feature-name
   # Create PR via GitHub UI
   ```

6. **Address review feedback**

7. **Merge to DEV after approval**

## Coding Standards

We follow these standards:

### Python Code Style

- **PEP 8** for Python code style
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

### Documentation

- Docstrings in **Google style**
- README files in Markdown
- Clear comments for complex logic

### Commit Messages

- Clear and descriptive
- Present tense ("Add feature" not "Added feature")
- Reference issue numbers when applicable

## Testing

### Test Types

- **Unit Tests:** Test individual functions and classes
- **Integration Tests:** Test component interactions
- **End-to-End Tests:** Test complete user flows

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=./ --cov-report=xml

# Run specific test file
pytest tests/test_backend.py

# Run specific test
pytest tests/test_backend.py::test_predict_endpoint_success
```

### Writing Tests

- Test files in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures for setup/teardown
- Mock external dependencies

## Docker Development

### Running with Docker Compose

```bash
# Start all services
docker-compose up

# Start specific service
docker-compose up frontend

# Rebuild containers
docker-compose up --build

# Run in background
docker-compose up -d
```

### Rebuilding Individual Services

```bash
# Rebuild and restart a specific service
docker-compose up --build backend
```

### Accessing Service Logs

```bash
# View logs for all services
docker-compose logs

# View logs for specific service
docker-compose logs frontend

# Follow logs
docker-compose logs -f
```

## Adding New Features

### Adding a New Service

1. **Create a new directory** for your service
2. **Add a Dockerfile** for containerization
3. **Add requirements.txt** for Python dependencies
4. **Implement your service code**
5. **Add to docker-compose.yml**
6. **Add tests** in the tests directory
7. **Update documentation**

### Extending Existing Services

1. **Identify the component** to extend
2. **Add your functionality** following existing patterns
3. **Add tests** for new functionality
4. **Update relevant documentation**

### Adding New ML Models

1. **Add model implementation** in the mlflow directory
2. **Define hyperparameter space** in param_grid_functions.py
3. **Add model to mlflow_setup.py**
4. **Add evaluation metrics** appropriate for the model
5. **Add tests** to validate model performance

## Troubleshooting

### Common Issues

1. **Docker Compose Errors:**
   - Check port conflicts
   - Ensure Docker daemon is running
   - Check service dependencies

2. **MLflow Connection Issues:**
   - Verify MLFLOW_TRACKING_URI
   - Check if MLflow service is running
   - Check network connectivity between services

3. **Model Training Failures:**
   - Check data availability
   - Verify sklearn version compatibility
   - Check for memory/resource constraints

4. **Test Failures:**
   - Run individual failed tests for more details
   - Check mocked dependencies
   - Verify environment setup

### Debugging Tips

1. **Use Visual Studio Code Debugger:**
   - Set breakpoints
   - Inspect variables
   - Step through code

2. **Add Logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   logging.debug("Variable value: %s", variable)
   ```

3. **Interactive Debugging in Docker:**
   ```bash
   docker-compose exec backend bash
   python -m pdb -c continue backend/main.py
   ```

4. **Check Container Logs:**
   ```bash
   docker-compose logs backend
   ```
