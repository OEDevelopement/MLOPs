import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Income Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 0.5rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
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
    
    race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
    is_White = 1 if race == "White" else 0
    
    nationality = st.radio("From USA?", ["Yes", "No"])
    from_USA = 1 if nationality == "Yes" else 0
    
    marital_status = st.selectbox(
        "Marital Status", 
        ["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
    )

with col2:
    st.markdown("<div class='sub-header'>Professional Information</div>", unsafe_allow_html=True)
    
    workclass = st.selectbox(
        "Work Class", 
        ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
    )
    
    educational_num = st.slider("Education Level (numeric)", min_value=1, max_value=16, value=10, 
                             help="1: No education, 16: Doctorate")
    
    occupation = st.selectbox(
        "Occupation", 
        ["Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Prof-specialty", "Other-service",
         "Sales", "Craft-repair", "Transport-moving", "Farming-fishing", "Machine-op-inspct", 
         "Tech-support", "Protective-serv", "Armed-Forces", "Priv-house-serv"]
    )
    
    relationship = st.selectbox(
        "Relationship", 
        ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"]
    )
    
    hours_per_week = st.slider("Hours per Week", min_value=1, max_value=100, value=40)
    
    gained_capital = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
    
    # Default value for prediction target (we'll include this in our model input)
    income_over_50K = 0

# Form submission
submit_button = st.button("Predict Income", type="primary", use_container_width=True)

# Display sample data visualization in the sidebar
with st.sidebar:
    st.header("Sample Data Insights")
    
    # Sample data for visualizations
    sample_data = {
        'Education': ['High School', 'Bachelors', 'Masters', 'Doctorate'],
        'Avg Income': [30000, 45000, 65000, 85000]
    }
    df_sample = pd.DataFrame(sample_data)
    
    st.subheader("Average Income by Education")
    fig = px.bar(df_sample, x='Education', y='Avg Income', color='Education')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Tips")
    st.info("""
    - Higher education levels often correlate with higher income
    - Working more hours per week generally increases income
    - Certain occupations like Executive and Professional roles have higher income potential
    """)
    
    st.markdown("---")
    st.caption("© 2025 Income Prediction App")

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
        "income >50K": income_over_50K,
        "is_White": is_White,
        "from_USA": from_USA,
        "gained-capital": gained_capital
    }
    
    # Create visualization of input data
    st.markdown("<div class='sub-header'>Your Profile</div>", unsafe_allow_html=True)
    
    # Display input data in a more visual way
    profile_cols = st.columns(4)
    with profile_cols[0]:
        st.metric("Age", age)
        st.metric("Education Level", educational_num)
    with profile_cols[1]:
        st.metric("Work Hours", hours_per_week)
        st.metric("Capital Gain", f"${gained_capital}")
    with profile_cols[2]:
        st.metric("Gender", "Male" if is_Male else "Female")
        st.metric("Race", "White" if is_White else "Other")
    with profile_cols[3]:
        st.metric("US Citizen", "Yes" if from_USA else "No")
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
            
            # Display prediction
            if prediction > 0.5:
                st.markdown("<div class='prediction-box prediction-high'>", unsafe_allow_html=True)
                st.markdown("### 🎉 Income Prediction: Above $50K")
                st.markdown(f"Confidence: {prediction:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='prediction-box prediction-low'>", unsafe_allow_html=True)
                st.markdown("### Income Prediction: Below $50K")
                st.markdown(f"Confidence: {(1-prediction):.2%}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            # Show factors that influenced prediction
            st.subheader("Key Factors")
            factors_cols = st.columns(2)
            
            # These are simplified example factors - in a real app you'd get these from the model
            with factors_cols[0]:
                st.markdown("#### Positive Factors")
                factors_pos = []
                if educational_num > 12:
                    factors_pos.append("Higher education level")
                if hours_per_week > 45:
                    factors_pos.append("Working more than 45 hours per week")
                if gained_capital > 0:
                    factors_pos.append("Having capital gains")
                if occupation in ["Exec-managerial", "Prof-specialty"]:
                    factors_pos.append("Working in a high-income occupation")
                
                if factors_pos:
                    for f in factors_pos:
                        st.markdown(f"✅ {f}")
                else:
                    st.markdown("No significant positive factors identified")
            
            with factors_cols[1]:
                st.markdown("#### Negative Factors")
                factors_neg = []
                if educational_num < 10:
                    factors_neg.append("Lower education level")
                if age < 25:
                    factors_neg.append("Younger age group")
                if marital_status in ["Never-married", "Divorced"]:
                    factors_neg.append("Marital status")
                if relationship in ["Own-child"]:
                    factors_neg.append("Dependent relationship")
                
                if factors_neg:
                    for f in factors_neg:
                        st.markdown(f"❌ {f}")
                else:
                    st.markdown("No significant negative factors identified")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")
            st.info("Make sure the FastAPI backend is running on http://localhost:8000")