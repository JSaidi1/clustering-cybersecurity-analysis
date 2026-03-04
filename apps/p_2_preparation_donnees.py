import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ================================================================================
#                         Partie 2 — Préparation des données 
# ================================================================================

# Travail demandé :

# 1. Sélectionner un sous-ensemble de variables numériques pertinentes.
# 2. Appliquer une normalisation.
# 3. Réduire éventuellement la dimension des données (PCA).

# Questions :

# * Pourquoi la réduction de dimension peut-elle améliorer le clustering ?
# * Quel pourcentage de variance doit être conservé ?

# -------------------------
# Variables importantes :

# * dur : durée de la connexion
# * proto : protocole réseau
# * sbytes : nombre d’octets envoyés par la source
# * dbytes : nombre d’octets envoyés par la destination
# * sttl : TTL de la source
# * dttl : TTL de la destination
# * spkts : nombre de paquets source
# * dpkts : nombre de paquets destination
# * rate : taux de transfert

# Chaque ligne correspond à **une connexion réseau**.

# --------------------------------------------------------------------------------
# Chemin des données 
# --------------------------------------------------------------------------------
# ───────────── INPUTS ─────────────

IN_DIR = os.path.join(".", "data")
IN_CSV = os.path.join(IN_DIR, "UNSW_NB15_training-set.csv")

# --------------------------------------------------------------------------------
# Chargement des données 
# --------------------------------------------------------------------------------
# --- Charger les données
df = pd.read_csv(IN_CSV)

# --- Apperçu des données
# print(df.head(5))

print("# --------------------------------------------------------------------------------")
print("# 1. Sélectionner un sous-ensemble de variables numériques pertinentes")
print("# --------------------------------------------------------------------------------")
features = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload", # débit
    "dload" # débit
]

X = df[features]

print("\n-X.head() :")
print(X.head())

print("\n-df.describe()")
print(df.describe())

print("# --------------------------------------------------------------------------------")
print("2. Appliquer une normalisation")
print("# --------------------------------------------------------------------------------")
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Conversion en DataFrame pour plus de lisibilité
X_scaled = pd.DataFrame(X_scaled, columns=features)

print("-X_scaled.head() :")
print(X_scaled.head())


print("# --------------------------------------------------------------------------------")
print("3. Réduire éventuellement la dimension des données (PCA)")
print("# --------------------------------------------------------------------------------")

# PCA pour conserver 95% de la variance
pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)

print("-Nombre de composantes :", pca.n_components_)

print("-les variables qui composent chaque composante :")
components = pd.DataFrame(
    pca.components_,
    columns=features
)

print(components)

print("# --------------------------------------------------------------------------------")
print("Q 1 - Pourquoi la réduction de dimension peut-elle améliorer le clustering ?")
print("# --------------------------------------------------------------------------------")
print("""
La réduction de dimension peut améliorer le clustering pour plusieurs raisons :

    1. Réduction du bruit
        => certaines variables contiennent peu d'information utile.
      
    2. Suppression des corrélations
        => certaines variables sont fortement corrélées (ex : spkts et sbytes).
      
    3. Distances plus fiables
        => dans des espaces de grande dimension, les distances deviennent moins discriminantes.
      
    4. Visualisation des clusters
        => PCA permet de projeter les données en 2 dimensions pour observer les clusters.
""")

print("# --------------------------------------------------------------------------------")
print("Q 2 - Quel pourcentage de variance doit être conservé ?")
print("# --------------------------------------------------------------------------------")
print("""
En pratique, on conserve généralement 90 % à 95 % de la variance totale.

Cela permet :

    - de garder la majorité de l'information du dataset
    - tout en réduisant significativement la dimension des données

(Dans ce TP, conserver 95 % de la variance est un choix approprié)
""")

print("Variance expliquée par chaque composante :")
print(pca.explained_variance_ratio_)

print("Variance totale conservée :")
print(sum(pca.explained_variance_ratio_))



