#!/usr/bin/env python
# coding: utf-8

# # I.Contexte et Objectif

# Prédire le départ des clients d'une banque à partir de leurs données personnelles et comportementales. Ce modèle permettra à la banque d’anticiper les risques d’attrition et de cibler les actions de fidélisation.

# # II. Problématique

# - Quels facteurs influencent le départ des clients ?
# 
# - Peut-on prédire à l’avance si un client va quitter la banque ?
# 
# 

# # III. Analyse exploratoire (EDA)

# In[1]:


#import fichier

import pandas as pd 

df = pd.read_csv("C:\\Users\\Valencia\\Downloads\\Churn_Modelling (1).csv")

#aperçu des 5 premières lignes 

df.head()


# In[2]:


# Dimensions du jeu de données
print(f"Nombre de lignes : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}")
 


# In[3]:


# Types de données et valeurs manquantes
df.info()


# Conclusion : Aucune valeur manquante dans mon Dataset

# In[4]:


df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])


# In[5]:


# Analyse de la variable cible 

import seaborn as sns
import matplotlib.pyplot as plt

# Répartition de la variable cible
sns.countplot(x="Exited", data=df)
plt.title("Répartition des clients quittant la banque (Exited)")
plt.xlabel("Exited (0 = reste, 1 = quitte)")
plt.ylabel("Nombre de clients")
plt.show()


# In[6]:


# Pourcentage de churn
churn_rate = df['Exited'].mean()
print(f"Taux de churn : {churn_rate:.2%}")


#  Le taux de churn de 20,37% signifie que 20,37% des clients du dataset ont quitté la banque, ce qui est une information clé dans un projet de classification sur le churn

# 📊 Interprétation
# Sur 10 000 clients, environ 2 037 ont quitté la banque (Exited = 1).
# 
# Les 79,63% restants, soit environ 7 963 clients, sont restés (Exited = 0).
# 
# Il s’agit d’un problème de classification binaire déséquilibré, mais pas fortement (en général, un déséquilibre commence à poser problème en dessous de 10–15%).

# # Analyse univariée + bivariée (relation avec Exited)
# 

# In[7]:


# Distribution de l'âge selon le churn
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='Age', hue='Exited', bins=30, kde=True, multiple="stack")
plt.title("Répartition de l'âge selon le churn")
plt.xlabel("Âge")
plt.ylabel("Nombre de clients")
plt.show()


# 📊 1. Age vs Exited
# Observation :
# 
# Les clients entre 30 et 40 ans sont majoritaires.
# 
# Le taux de churn augmente significativement avec l’âge, notamment après 45 ans.
# 
# Les plus jeunes clients partent peu.
# 
# Interprétation :
# 
# L’âge est un facteur discriminant fort.
# 
# Les clients plus âgés sont peut-être plus exigeants ou plus susceptibles de quitter la banque.
# 
# *** Pertinent à garder pour la modélisation.
# 
# 

# In[8]:


# Répartition des sexes par churn
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Gender', hue='Exited')
plt.title("Genre vs Churn")
plt.xlabel("Genre")
plt.ylabel("Nombre de clients")
plt.show()

# Taux de churn par genre
churn_by_gender = df.groupby('Gender')['Exited'].mean()
print(churn_by_gender)


# 2. Gender vs Exited
# Observation :
# 
# La proportion de clients hommes et femmes est équilibrée.
# 
# Le taux de churn est légèrement plus élevé chez les femmes.
# 
# Interprétation :
# 
# Le genre a un impact modéré mais existant.
# 
# Ce n’est pas une variable décisive seule, mais elle peut aider combinée à d’autres.
# 
# ***À garder dans le modèle, mais ne pas s’attendre à une importance très forte.

# In[9]:


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Geography', hue='Exited')
plt.title("Pays vs Churn")
plt.xlabel("Pays")
plt.ylabel("Nombre de clients")
plt.show()

# Taux de churn par pays
churn_by_country = df.groupby('Geography')['Exited'].mean()
print(churn_by_country)


# 3. Geography vs Exited
# Observation :
# 
# Les clients allemands ont un taux de churn nettement plus élevé.
# 
# Les clients espagnols et français quittent beaucoup moins la banque.
# 
# Interprétation :
# 
# La localisation géographique est très importante pour prédire le churn.
# 
# Il se peut que l’offre bancaire soit moins attractive ou le service client moins bon en Allemagne.
# 
# *** À garder absolument, très bonne variable explicative.

# In[18]:


plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='Balance', hue='Exited', bins=40, kde=True, multiple='stack')
plt.title("Solde (Balance) vs Churn")
plt.xlabel("Solde")
plt.ylabel("Nombre de clients")
plt.show()


# 4. Balance vs Exited
# Observation :
# 
# Il y a deux groupes très visibles : ceux avec balance = 0 et ceux avec un solde élevé.
# 
# Le churn est plus élevé dans les balances moyennes à élevées.
# 
# Interprétation :
# 
# Étonnamment, les clients avec 0 de solde partent peu.
# 
# Peut-être des comptes inactifs ou des comptes secondaires ?
# 
# Les clients avec plus d’argent sont plus à risque de partir (peut-être courtisés par la concurrence ?)
# 
# **** Variable utile, mais à interpréter finement.

# In[10]:


sns.countplot(data=df, x='NumOfProducts', hue='Exited')
plt.title("Nombre de produits vs Churn")
plt.xlabel("Nombre de produits bancaires")
plt.ylabel("Nombre de clients")
plt.show()


# 5. NumOfProducts vs Exited
# Observation :
# 
# Les clients ayant 1 produit sont les plus nombreux.
# 
# Les clients ayant 3 produits ou plus sont très rares, mais leur churn est élevé.
# 
# Les clients avec 4 produits ont un taux de churn extrêmement élevé.
# 
# Interprétation :
# 
# L’usage de plusieurs produits ne garantit pas la fidélité.
# 
# Un churn massif à 4 produits peut indiquer un segment à problème (VIP insatisfaits ? offres mal adaptées ?).
# 
# **** Très bon indicateur à inclure.
# 
# 

# In[11]:


plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='EstimatedSalary', hue='Exited', bins=40, kde=True, multiple='stack')
plt.title("Salaire estimé vs Churn")
plt.xlabel("Salaire")
plt.ylabel("Nombre de clients")
plt.show()


# EstimatedSalary vs Exited
# Observation :
# 
# La distribution est assez uniforme (salaire généré aléatoirement).
# 
# Le churn n’augmente pas clairement avec le salaire.
# 
# Interprétation :
# 
# Le salaire estimé n’est pas discriminant ici.
# 
# Peu corrélé à la probabilité de partir.
# 
# ⚠️ À tester en modélisation, mais probablement faible importance

# In[22]:


# Matrice de corrélation (numérique seulement)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corrélation entre variables numériques")
plt.show()


# Observation :
# 
# Age est positivement corrélé à Exited.
# 
# NumOfProducts a une corrélation négative.
# 
# CreditScore ou EstimatedSalary ont peu de corrélation directe.
# 
# Interprétation :
# 
# Certaines variables (comme Age) sont plus liées à la cible.
# 
# D’autres (comme CreditScore) peuvent jouer un rôle en interaction avec d’autres variables.
# 
# 

# In[12]:


# encodage 

#Gender → encodage binaire
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})


# In[13]:


#Geography → encodage One-Hot
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)


# Standardisation des variables numériques

# In[14]:


from sklearn.preprocessing import StandardScaler

# On ne standardise pas la variable cible 'Exited'
features_to_scale = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                     'NumOfProducts', 'EstimatedSalary']

scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])


# Séparation des features et de la target
# 

# In[15]:


X = df.drop('Exited', axis=1)
y = df['Exited']


# Split train/test

# In[16]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Entrainement du modèle : Regression Logistic

# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Initialisation du modèle (avec class_weight='balanced' pour gérer le déséquilibre)
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

# Entraînement
model.fit(X_train, y_train)

# Prédiction sur test
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Évaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))


# Precision (Précision) : parmi les clients prédits comme churners (classe 1), combien le sont vraiment ?
# 
# Classe 1 : 39% → Peu élevée, beaucoup de faux positifs.
# 
# Recall (Rappel ou Sensibilité) : parmi les clients qui ont réellement churné, combien ont été correctement détectés ?
# 
# Classe 1 : 70% → Bon rappel, tu détectes 70% des churners.
# 
# F1-score : compromis entre précision et rappel.
# 
# Classe 1 : 0.50 → Moyenne basse, signal que le modèle a du mal à bien équilibrer précision et rappel sur les churners.
# 
# Support : nombre d’échantillons par classe dans le test (1593 restés, 407 partis).
# 
# 

# Il détecte 70% des clients qui vont partir (rappel correct).
# 
# Il différencie bien les clients “restés” vs “partis” avec un AUC de 0.78.
# 
# La précision globale de 71% montre que le modèle n’est pas “au hasard”.
 
 

 #Je  peux prédire le churn avec une performance correcte.

#Mais je dois encore améliorer le modèle ou adapter ta stratégie selon ce que la banque préfère :

#Maximiser la détection des churners (favoriser le rappel)

#Minimiser les fausses alertes (favoriser la précision)
# # Le modèle actuel permet d’anticiper environ 70% des départs clients, mais génère aussi un nombre non négligeable de fausses alertes. Il constitue une bonne base pour affiner la stratégie de fidélisation, en ajustant le seuil de détection ou en intégrant des données supplémentaires pour améliorer la précision.

# Je vais essayer d'autres modèles 

# In[18]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Initialisation du modèle
rf_model = RandomForestClassifier(
    n_estimators=100,      # nombre d'arbres
    random_state=42,
    class_weight='balanced'  # pour gérer le déséquilibre
)
print("➡️ Début entraînement Random Forest")

# Entraînement
rf_model.fit(X_train, y_train)

# Prédiction
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Évaluation
print("Classification Report Random Forest:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix Random Forest:\n", confusion_matrix(y_test, y_pred_rf))
print("AUC-ROC Random Forest:", roc_auc_score(y_test, y_pred_proba_rf))


# Précision pour churners (1) : 78%, beaucoup mieux que la régression logistique.
# 
# Rappel pour churners (1) : 46%, plus faible qu'avant (régression logistique avait 70%).
# 
# F1-score plus élevé (0.58 vs 0.50), globalement meilleure performance sur la classe 1.
# 
# 2. Accuracy et AUC
# Accuracy = 86% (vs 71% LR) → grosse amélioration globale.
# 
# AUC-ROC = 0.85 (vs 0.78 LR) → meilleure capacité à distinguer churn/non-churn.
# 
#Interprétation: 

#Le Random Forest réduit drastiquement les fausses alertes → mieux pour ne pas gaspiller les ressources sur des clients fidèles.

#En revanche, il attrape moins de churners (rappel plus faible), donc il y a un compromis.

#Selon la stratégie de la banque (prévenir à tout prix ou éviter les faux positifs), tu peux choisir entre le modèle ou ajuster le seuil
# Ce que montre le modèle Random Forest :
# Le modèle est capable de bien repérer les clients qui vont rester : il les identifie correctement presque 97 fois sur 100 (très fiable).
# 
# Pour les clients qui risquent de partir, le modèle est plus prudent :
# 
# Quand il dit qu’un client va partir, il a raison 78% du temps (c’est fiable).
# 
# Mais il ne détecte que 46% des clients qui vont réellement partir (il en manque un peu plus de la moitié).
# 
# 
#Que cela signifie pour la banque ?
#La banque ne va pas trop gaspiller d'efforts à courir après des clients qui ne partiront pas (peu de fausses alertes).

#Par contre, beaucoup de clients à risque de départ ne seront pas détectés, donc la banque pourrait manquer des occasions de les retenir.
# Solution : essayer d'ajuster le modèle 
#     
#     Pour détecter plus de clients à risque (augmenter le rappel), on peut ajuster le seuil de décision du modèle Random Forest.
# 
# Explication rapide :
# Par défaut, un modèle classe un client en “partant” si la probabilité prédite est ≥ 0.5.
# Si on baisse ce seuil (par exemple à 0.3), le modèle devient plus “sensible” : il va prédire “partant” plus facilement, donc attraper plus de churners, mais aussi générer plus de fausses alertes.
# 
# 

# In[19]:


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Prédire les probabilités pour la classe "1" (partant)
y_proba = rf_model.predict_proba(X_test)[:, 1]

# Choisir un seuil plus bas, par exemple 0.3 au lieu de 0.5
threshold = 0.3
y_pred_adjusted = (y_proba >= threshold).astype(int)

# Évaluer les performances avec ce nouveau seuil
print("Classification Report avec seuil =", threshold)
print(classification_report(y_test, y_pred_adjusted))

print("Matrice de confusion avec seuil =", threshold)
print(confusion_matrix(y_test, y_pred_adjusted))


# Ce que j'ai amélioré en baissant le seuil à 0.3 :
# Je détecte maintenant 65 % des clients qui vont réellement partir (contre 46 % avant).
# 
# mon modèle est plus sensible aux clients à risque, ce qui est exactement ce que je cherchais.
# 
# L’accuracy globale reste bonne : 83 %.

# # 🧠 Conclusion :
# Mon modèle est maintenant plus utile pour anticiper les départs de clients, ce qui correspond exactement à mon objectif.
# 
# 📌 Ce modèle peut maintenant être utilisé comme outil d’aide à la décision marketing : cibler les 264 clients identifiés à risque pour les relancer, leur proposer des offres, etc.
# 
# 
import pickle

with open('random_forest_churn.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("➡️ Prêt à sauvegarder le modèle")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Modèle et scaler sauvegardés avec succès.")
