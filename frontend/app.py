import streamlit as st
import requests
import json
import os

# Backend URL konfigurieren
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://backend:8000')

st.title("Income Prediction App")

# Formular für die Vorhersage erstellen
st.header("Predict Income Level")
st.write("Enter the details below to predict if income exceeds $50K per year")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=38)
        workclass = st.selectbox("Work Class", options=[
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
            "Local-gov", "State-gov", "Without-pay", "Never-worked"
        ])
        education_num = st.slider("Education Level (1-16)", min_value=1, max_value=16, value=10)
        marital_status = st.selectbox("Marital Status", options=[
            "Married-civ-spouse", "Never-married", "Divorced", 
            "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
        ])
        occupation = st.selectbox("Occupation", options=[
            "Exec-managerial", "Prof-specialty", "Tech-support", "Adm-clerical",
            "Sales", "Craft-repair", "Transport-moving", "Handlers-cleaners",
            "Farming-fishing", "Machine-op-inspct", "Other-service", "Protective-serv", 
            "Armed-Forces", "Priv-house-serv"
        ])
    
    with col2:
        relationship = st.selectbox("Relationship", options=[
            "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried", "Other-relative"
        ])
        hours_per_week = st.slider("Hours per Week", min_value=1, max_value=100, value=40)
        is_male = st.radio("Gender", options=["Male", "Female"], horizontal=True)
        is_white = st.checkbox("Is White", value=True)
        from_usa = st.checkbox("From USA", value=True)
        gained_capital = st.number_input("Capital Gain ($)", min_value=0, value=0)
        
        # Dummy Feld für income (wird für die Vorhersage benötigt, aber mit 0 vorbelegt)
        income_over_50k = 0
    
    submit_button = st.form_submit_button("Predict Income")

if submit_button:
    # Daten für die API aufbereiten
    data = {
        "age": age,
        "workclass": workclass,
        "educational-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "hours-per-week": hours_per_week,
        "is_Male": 1 if is_male == "Male" else 0,
        "income >50K": income_over_50k,  # Dummy-Wert
        "is_White": 1 if is_white else 0,
        "from_USA": 1 if from_usa else 0,
        "gained-capital": gained_capital
    }
    
    with st.spinner("Calculating prediction..."):
        try:
            # Vorhersage vom Backend anfordern
            response = requests.post(f"{BACKEND_URL}/predict", json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                # Prüfen, ob wir den Placeholder nutzen
                if "is_placeholder" in result and result["is_placeholder"]:
                    st.warning("⚠️ The model is still in training. Please check back later for real predictions.")
                    st.info("This is a placeholder response while the actual model is being trained.")
                    
                    # Training status anzeigen wenn möglich
                    try:
                        mlflow_status = requests.get("http://mlflow:5000/api/2.0/mlflow/experiments/list", timeout=2)
                        if mlflow_status.status_code == 200:
                            st.success("MLflow server is running and training the model.")
                    except:
                        pass
                else:
                    # Echte Vorhersage anzeigen
                    prediction = result.get("predictions", [0])[0]
                    
                    # Ergebnis anzeigen
                    st.subheader("Prediction Result")
                    if prediction == 1:
                        st.success("The predicted income is OVER $50K per year")
                    else:
                        st.info("The predicted income is UNDER $50K per year")
                
                # Rohe Antwort anzeigen (für Debugging)
                with st.expander("Raw Response"):
                    st.json(result)
            else:
                st.error(f"Error making prediction: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend service. Please ensure backend is running.")
        except requests.exceptions.Timeout:
            st.error("Request timed out. Backend might be overloaded.")
        except Exception as e:
            if "HTTPConnectionPool" in str(e) and "Max retries exceeded" in str(e):
                st.error("⚠️ Backend service is unavailable. Please check if it's running.")
            elif "RemoteDisconnected" in str(e) or "Connection aborted" in str(e):
                st.error("⚠️ Connection to the backend service was interrupted.")
            else:
                st.error(f"An error occurred: {str(e)}")
            
            st.info("There might be an issue with the connection to the model service.")

# Training Status Information
st.sidebar.title("System Information")
try:
    # Prüfe, ob der MLflow-Server läuft
    mlflow_status = requests.get("http://mlflow:5000", timeout=2)
    st.sidebar.success("MLflow Server: Running")
    st.sidebar.info("Model training may be in progress.")
except:
    st.sidebar.warning("MLflow Server: Not accessible from frontend")

try:
    # Prüfe, ob der Model Service ein Placeholder ist
    model_health = requests.get("http://model_service:8080/health", timeout=2)
    if model_health.status_code == 200:
        health_data = model_health.json()
        if "status" in health_data and health_data["status"] == "placeholder":
            st.sidebar.warning("Model Service: Placeholder (Training in progress)")
        else:
            st.sidebar.success("Model Service: Trained model active")
except:
    st.sidebar.warning("Model Service: Not accessible from frontend")

# Footer
st.markdown("---")
st.markdown("Income Prediction App | Made with Streamlit & FastAPI")