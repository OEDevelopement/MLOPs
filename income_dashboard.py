import os
import pandas as pd
import streamlit as st
import numpy as np
import kagglehub as kh
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Caching, damit Daten und Modell nicht bei jedem Interaktionsschritt neu geladen bzw. trainiert werden
@st.cache_data
def load_data():
    # Datensatz herunterladen
    path = kh.dataset_download("wenruliu/adult-income-dataset")
    files = os.listdir(path)
    # st.write("Dateien im Verzeichnis:", files)
    # Erste CSV-Datei auswählen
    csv_file = [f for f in files if f.endswith('.csv')][0]
    csv_path = os.path.join(path, csv_file)
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource
def train_model(df):
    # Zielvariable ('income') und Features festlegen
    target = 'income'
    features = [col for col in df.columns if col != target]
    
    # Spalten entfernen
    if 'fnlwgt' in features:
        features.remove('fnlwgt')
    if 'educational-num' in features:
        features.remove('educational-num')
    
    X = df[features]
    y = df[target]
    
    # Annahme: Numerische Spalten besitzen numerische Datentypen, ansonsten werden sie als kategorial angesehen
    numeric_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    
    # Numerische Features skalieren, kategoriale Features mittels One-Hot-Encoding umwandeln
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X, y)
    return pipeline, features, numeric_cols, categorical_cols, X

# Hauptteil der App
st.title("Einkommensvorhersage Dashboard")
st.write("Dieses Dashboard sagt anhand von Eingabedaten voraus, ob das Einkommen über oder unter 50.000 liegt.")

# Laden des Datensatzes und trainieren des Modells
df = load_data()
model, feature_names, numeric_cols, categorical_cols, X_data = train_model(df)

st.header("Geben Sie Ihre Daten ein")
st.write("Bitte füllen Sie alle Felder aus. Bei kategorialen Merkmalen können nur die im Datensatz vorhandenen Optionen gewählt werden.")

# Erzeuge Eingabeformular
user_input = {}
for col in feature_names:
    if col in numeric_cols:
        # Für numerische Features: Standardwerte aus dem Datensatz (Mittelwert, Minimum, Maximum)
        min_val = float(X_data[col].min())
        max_val = float(X_data[col].max())
        mean_val = float(X_data[col].mean())
        # Wenn die Spalte "age" heißt, soll der Input ganzzahlig erfolgen
        if col.lower() == 'age':
            user_input[col] = st.number_input(
                label=f"{col}",
                min_value=int(min_val),
                max_value=int(max_val),
                value=int(mean_val),
                step=1
            )
        else:
            user_input[col] = st.number_input(
                label=f"{col}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )
    else:
        # Für kategoriale Features: Optionen aus dem Datensatz
        options = sorted(X_data[col].unique())
        user_input[col] = st.selectbox(f"{col}", options)

# Zeige die eingegebenen Daten an
st.subheader("Ihre Eingaben")
input_df = pd.DataFrame([user_input])
st.write(input_df)

# Vorhersage-Button
if st.button("Einkommen vorhersagen"):
    prediction = model.predict(input_df)
    st.subheader("Vorhersage")
    st.write(f"Die Vorhersage lautet: **{prediction[0]}**")



##########
# In Ordner Terminal öffnen
# Streamlit starten mit: streamlit run income_dashboard.py
# http://localhost:8501
##########