import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Chargement du modèle et du scaler ---
model = pickle.load(open("random_forest_churn.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("📉 Prédiction du Churn Bancaire")

st.sidebar.header("📋 Paramètres Client")

def get_user_input():
    credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)
    age = st.sidebar.slider("Âge", 18, 92, 40)
    tenure = st.sidebar.slider("Ancienneté (années)", 0, 10, 3)
    balance = st.sidebar.number_input("Solde (€)", 0.0, 250000.0, 50000.0)
    num_products = st.sidebar.selectbox("Nombre de produits", [1, 2, 3, 4])
    has_cr_card = st.sidebar.selectbox("Carte de crédit ?", [1, 0])
    is_active_member = st.sidebar.selectbox("Client actif ?", [1, 0])
    estimated_salary = st.sidebar.number_input("Salaire estimé (€)", 0.0, 200000.0, 50000.0)
    gender = st.sidebar.selectbox("Genre", ["Male", "Female"])
    geography = st.sidebar.selectbox("Pays", ["France", "Spain", "Germany"])

    gender_val = 0 if gender == "Male" else 1
    geography_spain = 1 if geography == "Spain" else 0
    geography_germany = 1 if geography == "Germany" else 0

    # Données brutes
    data = {
        'CreditScore': credit_score,
        'Gender': gender_val,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
        'Geography_Germany': geography_germany,
        'Geography_Spain': geography_spain
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = get_user_input()

# --- Préparation des données ---
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# --- Prédiction ---
proba = model.predict_proba(input_df)[0][1]
pred = int(proba >= 0.3)  # Seuil personnalisé

# --- Affichage ---
st.subheader("🔍 Résultat")
st.write("**Probabilité de churn :**", f"{proba:.2%}")

if pred == 1:
    st.error("⚠️ Ce client est **à risque** de départ.")
else:
    st.success("✅ Ce client est **fidèle**.")

st.write("---")
st.write("📊 Données utilisées pour la prédiction :")
st.write(input_df)
