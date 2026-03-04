import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("UNSW_NB15_testing-set.csv")

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
relevant_cols = [
        'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 
        'spkts', 'dpkts', 'rate'
    ]
def select_cyber_features(df):
    available_cols = [c for c in relevant_cols if c in df.columns]
    features = df[available_cols].copy()

    if features.isnull().values.any():
        print("⚠️ Nettoyage : Valeurs manquantes détectées et remplacées par 0.")
        features = features.fillna(0) 

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    print(f"Features sélectionnées et nettoyées : {available_cols}")
    return features, X_scaled



# 3. Analyser les distributions des variables.
def analyze_distributions(df, cols):
    n_cols = 2
    n_rows = (len(cols) + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], bins=30, color='skyblue')
        axes[i].set_title(f'Distribution de {col}')
        axes[i].set_yscale('log')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()



# 4. Détecter les valeurs extrêmes.
def check_outliers(df, cols):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[cols])
    plt.yscale('log') 
    plt.title("Détection visuelle des outliers (Échelle Log)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Questions :

# * Pourquoi certaines variables doivent-elles être normalisées ?
# Parce que les données ne sont pas notées de la même manière (durée peut être de 0.00001 et sbytes peut atteindre 1000000).
# Il faut donc normaliser pour tout mettre à la même échelle. 

# * Pourquoi certaines variables catégorielles doivent être encodées ?
# Parce que les ML ne comprennent que les nombres. 
# L'encodage transforme des catégories string en valeurs numériques. 


explore_structure(df)
missing_info = analyse_missing_data(df)
print(missing_info)
features_df, X_scaled = select_cyber_features(df)
analyze_distributions(df, relevant_cols)
check_outliers(df, relevant_cols)


if __name__ == "__main__":
    df = pd.read_csv("UNSW_NB15_testing-set.csv")
    explore_structure(df)
