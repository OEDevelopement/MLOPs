import streamlit as st
import requests

st.title("Mein Frontend")

# Beispielhafter Aufruf des Backend
response = requests.get("http://backend:8000/")
st.write("Backend-Antwort:", response.text)
