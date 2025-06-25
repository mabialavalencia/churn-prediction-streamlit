# 🔍 Prédiction de Churn Bancaire avec Machine Learning et Streamlit

Ce projet vise à prédire le **churn** (le départ d’un client) dans le secteur bancaire à partir de données clients.  
Il combine une **analyse exploratoire**, un **modèle de machine learning** (Random Forest) et une **application web interactive** développée avec **Streamlit**.

---

## 📈 Contexte

Les banques souhaitent identifier les clients à risque de départ afin de **mieux cibler leurs actions de fidélisation** (offres commerciales, relances, appels sortants...).

Anticiper le churn permet :
- D’optimiser les ressources marketing
- De réduire les pertes de revenus
- D’améliorer l'expérience client

---

## 🎯 Objectif

Créer une application permettant :
- D’analyser les variables influençant le churn
- D’entraîner un modèle prédictif fiable
- De **prédire en temps réel** si un client est à risque, via une interface Streamlit

---

## 🔬 Données utilisées

Fichier : `Churn_Modelling.csv`  
- **10 000 clients** d’une banque fictive  
- Variables disponibles :
  - Âge, ancienneté, solde bancaire, produits utilisés, etc.
  - Pays, genre, statut actif, etc.
  - Cible : `Exited` (1 = client parti, 0 = client resté)

---

## 🧠 Pipeline de Modélisation

1. **Analyse exploratoire des données (EDA)**
2. Nettoyage & encodage (OneHot + Label)
3. Standardisation des variables numériques
4. Séparation train/test
5. Entraînement d’un modèle `RandomForestClassifier`
6. Ajustement du seuil de classification (0.3 au lieu de 0.5) pour **favoriser le rappel**
7. Sauvegarde du modèle (`.pkl`) et du scaler

---

## 💻 Application Streamlit

L’utilisateur peut :
- Entrer les données d’un client
- Obtenir une **probabilité de churn**
- Recevoir une **interprétation simple** : "Client à risque" ou "Client fidèle"

> Application locale lancée via :  
```bash
streamlit run app.py

