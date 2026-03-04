import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# # Partie 1 — Exploration des données
# Objectifs :
# 1. Comprendre la structure du dataset.
def explore_structure(df):
    print("\n--- Structure du Dataset ---")
    print(df.info())
    
    print("\n--- Statistiques Clés ---")
    print(df.describe())


    
def analyse_missing_data(df) :
    print("\n--- Valeurs manquantes ---")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)

    missing_table = pd.DataFrame({
        "Colonnes" : missing.index,
        "Valeurs manquantes" : missing.values,
        "Pourcentage" : missing_pct.values
    })
    missing_table = missing_table[missing_table["Valeurs manquantes"] > 0]. sort_values("Pourcentage", ascending=False)
    return missing_table



# 2. Identifier les variables pertinentes pour le clustering.
def select_cyber_features(df):
    relevant_cols = [
        'dur', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 
        'spkts', 'dpkts', 'rate'
    ]
    
    available_cols = [c for c in relevant_cols if c in df.columns]
    

    features = df[available_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    print(f"Features sélectionnées : {available_cols}")
    return features, X_scaled



# 3. Analyser les distributions des variables.
def analyze_distributions(df, columns):
    n_cols = 2
    n_rows = (len(columns) + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i], bins=30, color='skyblue')
        axes[i].set_title(f'Distribution de {col}')
        axes[i].set_yscale('log')
    plt.tight_layout()
    plt.show()



# 4. Détecter les valeurs extrêmes.
def check_outliers(df, columns):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[columns])
    plt.yscale('log') 
    plt.title("Détection visuelle des outliers (Échelle Log)")
    plt.xticks(rotation=45)
    plt.show()

# Questions :

# * Pourquoi certaines variables doivent-elles être normalisées ?
# Parce que les données ne sont pas notées de la même manière (durée peut être de 0.00001 et sbytes peut atteindre 1000000).
# Il faut donc normaliser pour tout mettre à la même échelle. 

# * Pourquoi certaines variables catégorielles doivent être encodées ?
# Parce que les ML ne comprennent que les nombres. 
# L'encodage transforme des catégories string en valeurs numériques. 


