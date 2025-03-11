import streamlit as st
import requests
import pandas as pd
import json
import os
import time
import socket
from threading import Thread
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY, CollectorRegistry
from http.server import HTTPServer, BaseHTTPRequestHandler

# Create a custom registry to avoid conflicts with the default registry
custom_registry = CollectorRegistry()

# Create metrics using the custom registry
PAGE_VIEWS = Counter('frontend_page_views_total', 'Total number of page views', registry=custom_registry)
PREDICTIONS = Counter('frontend_predictions_total', 'Total number of predictions made', registry=custom_registry)
FRONTEND_UP = Gauge('frontend_up', 'Status of the frontend service', registry=custom_registry)

# Simple HTTP handler for Prometheus metrics
class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', CONTENT_TYPE_LATEST)
        self.end_headers()
        self.wfile.write(generate_latest(custom_registry))
    
    # Disable logging to console
    def log_message(self, format, *args):
        return

# Start a simple metrics server in a separate thread
def run_metrics_server(port=8001):
    try:
        # Set the frontend as up
        FRONTEND_UP.set(1)
        
        # Create and start the server
        server = HTTPServer(('0.0.0.0', port), MetricsHandler)
        print(f"Starting metrics server on port {port}")
        server.serve_forever()
    except Exception as e:
        print(f"Error starting metrics server: {e}")

# Start the metrics server if not already running
if 'metrics_server_started' not in st.session_state:
    # Try to check if the port is already in use
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    in_use = False
    try:
        sock.bind(('0.0.0.0', 8001))
    except:
        in_use = True
    finally:
        sock.close()
    
    if not in_use:
        # Start the metrics server in a background thread
        metrics_thread = Thread(target=run_metrics_server, daemon=True)
        metrics_thread.start()
    
    st.session_state.metrics_server_started = True

# Count a page view when the app loads (only once per session)
if 'page_viewed' not in st.session_state:
    PAGE_VIEWS.inc()
    st.session_state.page_viewed = True

def create_monitoring_page():
    st.title("Simple System Monitoring")
    
    # Refresh button
    if st.button("Refresh Data"):
        st.experimental_rerun()
    
    # Try to get metrics from backend
    try:
        backend_url = os.environ.get('BACKEND_URL', 'http://backend:8000')
        response = requests.get(f"{backend_url}/metrics", timeout=2)
        
        if response.status_code == 200:
            metrics_text = response.text
            
            # Extract and display key metrics
            api_calls = extract_metric(metrics_text, "api_calls_total")
            predictions = extract_metric(metrics_text, "predictions_total")
            errors = extract_metric(metrics_text, "prediction_errors_total")
            active = extract_metric(metrics_text, "active_requests")
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total API Calls", api_calls)
                st.metric("Active Requests", active)
            
            with col2:
                st.metric("Total Predictions", predictions)
                st.metric("Prediction Errors", errors)
                
            # Calculate success rate
            if predictions and float(predictions) > 0:
                success_rate = (float(predictions) - float(errors or 0)) / float(predictions) * 100
                st.progress(min(success_rate/100, 1.0))
                st.text(f"Prediction Success Rate: {success_rate:.1f}%")
            
            # Raw metrics (collapsed)
            with st.expander("Raw Metrics"):
                st.code(metrics_text)
        else:
            st.error(f"Failed to get metrics: HTTP {response.status_code}")
    
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        st.info("Make sure the FastAPI backend is running and the /metrics endpoint is available")

def extract_metric(metrics_text, metric_name):
    """Extract the value of a metric from the raw prometheus text"""
    for line in metrics_text.split('\n'):
        if line.startswith(metric_name) and not line.startswith(f"{metric_name}_"):
            return line.split(' ')[1]
    return "0"

# Page configuration
if 'page_config_set' not in st.session_state:
    st.set_page_config(
        page_title="Income Prediction App",
        page_icon="💰",
        layout="centered"
    )
    st.session_state.page_config_set = True

# Navigation selection
if st.sidebar.selectbox("Navigation", ["Income Prediction", "Monitoring"]) == "Monitoring":
    create_monitoring_page()
else:
    # Custom CSS for a fancy look and smaller overall layout
    st.markdown("""
    <style>
        /* Gradient background for the entire page */
        body {
            background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
        }
        /* Constrain the width of the main container */
        .block-container {
            max-width: 800px;
            margin: auto;
            padding-top: 2rem;
        }
        /* Header styling */
        .main-header {
            font-size: 2rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .sub-header {
            font-size: 1.25rem;
            color: #0D47A1;
            margin-bottom: 0.5rem;
        }
        /* Prediction box styling with hover effect */
        .prediction-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s;
        }
        .prediction-box:hover {
            transform: scale(1.03);
        }
        .prediction-high {
            background-color: #C8E6C9;
            border: 2px solid #4CAF50;
        }
        .prediction-low {
            background-color: #FFCDD2;
            border: 2px solid #F44336;
        }
    </style>
    """, unsafe_allow_html=True)

    # API endpoint
    API_URL = os.environ.get('API_URL','http://backend:8000/predict')

    # Page title
    st.markdown("<h1 class='main-header'>Income Prediction Application</h1>", unsafe_allow_html=True)
    st.markdown("### Predict whether someone's income exceeds $50K based on census data")

    # Create two columns for the form
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='sub-header'>Personal Information</div>", unsafe_allow_html=True)
        
        age = st.slider("Age", min_value=18, max_value=90, value=35)
        
        gender = st.radio("Gender", ["Male", "Female"])
        is_Male = 1 if gender == "Male" else 0
        
        race = st.radio("Race", ["White", "Other"])
        is_White = 1 if race == "White" else 0
        
        nationality = st.radio("From USA?", ["Yes", "No"])
        from_USA = 1 if nationality == "Yes" else 0
        
        marital_status = st.selectbox(
            "Marital Status", 
            ["Married", "Never-Married", "Widowed/Separated"]
        )

    with col2:
        st.markdown("<div class='sub-header'>Professional Information</div>", unsafe_allow_html=True)
        
        workclass = st.selectbox(
            "Work Class", 
            ['Private', 'Government', 'Self-Employed', 'Unemployed']
        )
        
        educational_num = st.slider("Education Level (years)", min_value=1, max_value=16, value=10, 
                                help="1: No education, 16: Doctorate")
        
        occupation = st.selectbox(
            "Occupation", 
            ['Simple-Services', 'Professional', 'Public Safety',
            'Specialized-Services', 'Administrative', 'Management', 'Sales']
        )
        
        relationship = st.selectbox(
            "Relationship", 
            ['Husband', 'Wife', 'Child', 'Shared-Housing', 'Single']
        )
        
        hours_per_week = st.slider("Hours per Week", min_value=1, max_value=100, value=40)
        
        gained_capital = st.radio("Capital Gain", ["Yes", "No"])
        gained_capital_int = 1 if gained_capital == "Yes" else 0
        
    # Form submission
    submit_button = st.button("Predict Income", type="primary", use_container_width=True)

    # Process prediction when form is submitted
    if submit_button:
        # Increment prediction counter when a prediction is made
        PREDICTIONS.inc()
        
        # Prepare data for API
        data = {
            "age": age,
            "workclass": workclass,
            "educational-num": educational_num,
            "marital-status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "hours-per-week": hours_per_week,
            "is_Male": is_Male,
            "is_White": is_White,
            "from_USA": from_USA,
            "gained-capital": gained_capital_int
        }
        
        # Create visualization of input data
        st.markdown("<div class='sub-header'>Your Profile</div>", unsafe_allow_html=True)
        
        # Display input data in a more visual way
        profile_cols = st.columns(4)
        with profile_cols[0]:
            st.metric("Age", age)
            st.metric("Education Years", educational_num)
        with profile_cols[1]:
            st.metric("Work Hours", hours_per_week)
            st.metric("Capital Gain", "Yes" if gained_capital_int == 1 else "No")
        with profile_cols[2]:
            st.metric("Gender", "Male" if is_Male == 1 else "Female")
            st.metric("Race", "White" if is_White == 1 else "Other")
        with profile_cols[3]:
            st.metric("US Citizen", "Yes" if from_USA == 1 else "No")
            st.metric("Marital Status", marital_status)
            
        # Show loading spinner while making the prediction
        with st.spinner("Predicting income..."):
            try:
                # Make API request
                response = requests.post(API_URL, json=data)
                response.raise_for_status()
                
                # Get prediction result
                result = response.json()
                prediction = result.get("predictions", [0])[0]
                
                # Display prediction with a fancy box
                if prediction > 0.5:
                    st.markdown("<div class='prediction-box prediction-high'>", unsafe_allow_html=True)
                    st.markdown("### 🎉 Income Prediction: Above $50K")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='prediction-box prediction-low'>", unsafe_allow_html=True)
                    st.markdown("### Income Prediction: Below $50K")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")
                st.info("Make sure the FastAPI backend is running on http://backend:8000")