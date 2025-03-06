import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px

# Set page config to a centered layout without a sidebar
st.set_page_config(
    page_title="Income Prediction App",
    page_icon="💰",
    layout="centered"
)

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
API_URL = "http://backend:8000/predict"

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
            st.info("Make sure the FastAPI backend is running on http://localhost:8000")
