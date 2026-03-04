from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # Partie 3 — Application des méthodes de clustering
# Chaque groupe doit appliquer les algorithmes suivants :

# ### K-Means
# * déterminer le nombre de clusters optimal
def find_optimal_clusters(X_scaled, max_k=10):
    print("Nombre de clusters optimal.")
    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
    
        inertias.append(kmeans.inertia_)
        
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouettes.append(score)
        print(f"k={k} terminé.")


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_title('Méthode du Coude (Inertie)')
    ax1.set_xlabel('Nombre de clusters (k)')
    ax1.set_ylabel('Inertie')

    ax2.plot(k_range, silhouettes, 'ro-')
    ax2.set_title('Score de Silhouette')
    ax2.set_xlabel('Nombre de clusters (k)')
    ax2.set_ylabel('Score')

    plt.tight_layout()
    plt.show()

    best_k = k_range[silhouettes.index(max(silhouettes))]
    return best_k


# * analyser les centres de clusters
def analyze_cluster_centers(kmeans_model, feature_names, scaler):
    centers_scaled = kmeans_model.cluster_centers_
    centers_real = scaler.inverse_transform(centers_scaled[:, :len(feature_names)])
    
    df_centers = pd.DataFrame(centers_real, columns=feature_names)
    df_centers.index.name = 'Cluster'
    
    print("\n--- Profils types des Clusters (Valeurs réelles) ---")
    print(df_centers)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_centers, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Signature des Clusters (Centroïdes)")
    plt.show()
    
    return df_centers


# ### Clustering hiérarchique
# * construire un dendrogramm
def plot_dendrogram(Z, p=12):
    """Affiche le dendrogramme à partir d'une matrice de liaison Z."""
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode='lastp', p=p, show_contracted=True)
    plt.title("Dendrogramme du trafic réseau")
    plt.ylabel("Distance de Ward")
    plt.show()


# * déterminer un nombre de clusters pertinent
def get_hca_clusters(X_scaled, n_clusters=None, sample_size=1000):
    """
    Calcule les liens et retourne les labels. 
    Utilise plot_dendrogram pour le visuel.
    """
    indices = np.random.choice(X_scaled.shape[0], min(sample_size, X_scaled.shape[0]), replace=False)
    X_sample = X_scaled[indices]

    Z = linkage(X_sample, method='ward')

    plot_dendrogram(Z)

    labels = None
    if n_clusters:
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    return labels, indices



# ### DBSCAN
# * choisir les paramètres epsilon et min_samples
def find_best_epsilon(X_scaled, k=5):
    """
    Affiche le graphique des distances aux k-plus-proches voisins.
    Le 'coude' du graphique indique la valeur idéale de epsilon.
    """
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)

    distances = np.sort(distances[:, k-1], axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f"Graphique de la K-distance (k={k})")
    plt.xlabel("Points triés par distance")
    plt.ylabel(f"Distance au {k}-ième voisin (Epsilon)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return distances



# * identifier les points considérés comme bruit
def isolate_noise_points(df_original, labels):
    """
    Isole les points considérés comme du bruit par DBSCAN.
    """
    df_analysis = df_original.copy()
    df_analysis['Cluster'] = labels
    
    noise_df = df_analysis[df_analysis['Cluster'] == -1].copy()
    
    print(f"Nombre de points de bruit identifiés : {len(noise_df)}")
    return noise_df