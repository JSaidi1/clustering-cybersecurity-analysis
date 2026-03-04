from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, dbscan
from sklearn.decomposition import PCA
from . import p_1_exploration_donnees
from . import p_2_preparation_donnees
from . import p_3_clustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

print("# ================================================================================")
print("#                         Partie 4 — Comparaison des méthodes") 
print("# ================================================================================")

print("# --------------------------------------------------------------------------------")
print("# Nombre de clusters obtenus")
print("# --------------------------------------------------------------------------------")
# Entraîner KMeans avec k choisi + profils
best_k = p_3_clustering.find_optimal_clusters(p_2_preparation_donnees.X_scaled, max_k=10)
kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=0)
clusters_kmeans = kmeans.fit_predict(p_2_preparation_donnees.X_pca)

# Nombre de clusters KMeans
n_clusters_kmeans = p_1_exploration_donnees.find_optimal_clusters(p_2_preparation_donnees.X_scaled, max_k=10)

# ---
best_k_ward = 5
ward = AgglomerativeClustering(n_clusters=best_k_ward, linkage="ward")
clusters_ward = ward.fit_predict(p_2_preparation_donnees.X_pca)

# Nombre de clusters Ward
n_clusters_ward = len(set(clusters_ward))

# Nombre de clusters DBSCAN (sans le bruit)
clusters_dbscan = dbscan.fit_predict(p_2_preparation_donnees.X_pca)
n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)

print("Clusters KMeans :", n_clusters_kmeans)
print("Clusters Ward :", n_clusters_ward)
print("Clusters DBSCAN :", n_clusters_dbscan)

print("# --------------------------------------------------------------------------------")
print("Structure des clusters")
print("# --------------------------------------------------------------------------------")

# PCA pour visualisation
pca_vis = PCA(n_components=2)
X_vis = pca_vis.fit_transform(p_2_preparation_donnees.X_scaled)

# KMeans
plt.scatter(X_vis[:,0], X_vis[:,1], c=clusters_kmeans, cmap='viridis')
plt.title("Structure des clusters - KMeans")
plt.show()

# Ward
plt.scatter(X_vis[:,0], X_vis[:,1], c=clusters_ward, cmap='plasma')
plt.title("Structure des clusters - Ward")
plt.show()

# DBSCAN
plt.scatter(X_vis[:,0], X_vis[:,1], c=clusters_dbscan, cmap='coolwarm')
plt.title("Structure des clusters - DBSCAN")
plt.show()

print("# --------------------------------------------------------------------------------")
print("Cohérence (silhouette) + distances intra/inter")
print("# --------------------------------------------------------------------------------")

from sklearn.metrics import silhouette_score, pairwise_distances
import numpy as np

def n_clusters(labels):
    s = set(labels)
    return len(s) - (1 if -1 in s else 0)

print("KMeans silhouette:", silhouette_score(p_2_preparation_donnees.X_pca, clusters_kmeans))
print("Ward   silhouette:", silhouette_score(p_2_preparation_donnees.X_pca, clusters_ward))

if n_clusters(clusters_dbscan) >= 2:
    print("DBSCAN silhouette:", silhouette_score(p_2_preparation_donnees.X_pca, clusters_dbscan))
else:
    print("DBSCAN silhouette: non fiable (trop peu de clusters)")

# intra KMeans = inertie
print("KMeans intra (inertie):", kmeans.inertia_)

# inter KMeans = distances entre centres
centers = kmeans.cluster_centers_
print("KMeans inter (centres):\n", pairwise_distances(centers))


print("# --------------------------------------------------------------------------------")
print("Sensibilité aux paramètres")
print("# --------------------------------------------------------------------------------")

for eps_test in [0.3, 0.5, 0.7, 1.0]:
    labels = DBSCAN(eps=eps_test, min_samples=10).fit_predict(p_2_preparation_donnees.X_pca)
    print("eps:", eps_test, "| clusters:", n_clusters(labels), "| bruit%:", (labels==-1).mean())