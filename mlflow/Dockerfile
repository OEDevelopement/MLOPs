FROM python:3.9-slim  

WORKDIR /mlflow_app  
RUN mkdir -p data/processed/  
# Copy the serve_model.sh script into the container  
COPY serve_model.sh .  
COPY mlflow_setup.py .
COPY param_grid_functions.py .
COPY model_validation.py .
COPY ./data/processed/processed_data.csv ./data/processed/

# Make the serve_model.sh script executable  
RUN chmod +x serve_model.sh  

RUN chmod +x mlflow_setup.py
RUN chmod +x param_grid_functions.py
RUN chmod +x model_validation.py

# Install MLflow and scikit-learn  
RUN pip install --no-cache-dir mlflow scikit-learn  

# Expose the necessary ports for MLflow  
EXPOSE 5000  
EXPOSE 8080  

# Use the correct path to the script
CMD ["./serve_model.sh"]