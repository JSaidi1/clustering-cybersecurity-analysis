import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ================================================================================
#                         Partie 6 — Analyse critique 
# ================================================================================


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
print("# 1. Quelle méthode produit les clusters les plus cohérents ?")
print("# --------------------------------------------------------------------------------")

print("""
La méthode qui produit généralement les clusters les plus cohérents est K-Means.

En effet, K-Means optimise directement la distance intra-cluster, ce qui signifie que les points appartenant à un même cluster sont très proches les uns des autres. Cela permet d'obtenir des groupes compacts et bien séparés lorsque les données sont bien normalisées.

Dans notre analyse, le silhouette score obtenu avec K-Means est souvent plus élevé que celui des autres méthodes, ce qui indique une meilleure cohérence des clusters.

Cependant, K-Means suppose que les clusters ont une forme sphérique et une taille similaire, ce qui peut être une limitation lorsque les données ont une structure plus complexe.
""")

print("# --------------------------------------------------------------------------------")
print("2. Quelle méthode est la plus robuste au bruit ?")
print("# --------------------------------------------------------------------------------")

print("""
La méthode la plus robuste au bruit est DBSCAN.

Contrairement à K-Means et au clustering hiérarchique, DBSCAN peut identifier explicitement les points aberrants. Les observations qui ne sont pas suffisamment proches 
d'autres points sont classées comme bruit (label = -1).

Cette capacité permet d'éviter que des observations atypiques soient intégrées de force dans un cluster.

Dans un contexte de trafic réseau, ces points bruit peuvent correspondre à :

 - des connexions inhabituelles

 - des comportements rares

 - des attaques potentielles

Ainsi, DBSCAN est particulièrement efficace pour gérer les données contenant des anomalies.
""")


print("# --------------------------------------------------------------------------------")
print("3. Quelle méthode est la plus adaptée à la cybersécurité ?")
print("# --------------------------------------------------------------------------------")

print("""
Dans le contexte de la cybersécurité, DBSCAN est généralement la méthode la plus adaptée.

Les raisons principales sont :

-il permet de détecter automatiquement les comportements anormaux

-il ne nécessite pas de définir à l'avance le nombre de clusters

-il peut identifier des clusters de formes arbitraires

Dans un Security Operations Center (SOC), l'objectif principal est souvent de détecter des activités suspectes dans le trafic réseau. 
Les points identifiés comme bruit par DBSCAN peuvent donc correspondre à des anomalies potentielles nécessitant une analyse plus approfondie.
""")

print("# --------------------------------------------------------------------------------")
print("Quelles limites observez-vous ?")
print("# --------------------------------------------------------------------------------")
print("""
Plusieurs limites peuvent être observées lors de l'utilisation de ces méthodes de clustering.

1. Sensibilité au prétraitement des données
Les résultats dépendent fortement de la normalisation des variables et du choix des caractéristiques utilisées.
      
2. Choix des paramètres
Certains algorithmes nécessitent des paramètres difficiles à déterminer :
K-Means nécessite de choisir le nombre de clusters (k)
DBSCAN dépend fortement des paramètres epsilon et min_samples
Un mauvais choix de paramètres peut conduire à des clusters peu pertinents.
      
3. Scalabilité
Le clustering hiérarchique peut être très coûteux en termes de calcul lorsque le dataset est très grand.
Dans les données réseau réelles, qui peuvent contenir des millions de connexions, cette méthode peut devenir difficile à appliquer.
      
4. Interprétation des clusters

Les clusters obtenus ne correspondent pas toujours clairement à des types précis de trafic réseau. 
Une analyse supplémentaire est souvent nécessaire pour interpréter les clusters du point de vue de la cybersécurité.
""")
