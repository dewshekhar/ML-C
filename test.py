import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from the provided GitHub link
url = "https://raw.githubusercontent.com/dewshekhar/DFS-and-BFS-for-8-Puzzel/main/cancer.csv"
data = pd.read_csv(url)

# Drop non-numeric columns and constant columns
data_numeric = data.select_dtypes(include=[np.number])
data_numeric = data_numeric.loc[:, ~data_numeric.columns.duplicated()]
data_numeric = data_numeric.dropna(axis=1, how='all')  # Drop columns with all missing values

# Handle missing values using mean imputation
data_imputed = data_numeric.fillna(data_numeric.mean())

# Standardize the data
data_scaled = (data_imputed - data_imputed.mean()) / data_imputed.std()

# Number of clusters
num_clusters = 2

# Manual K-Means clustering
def k_means(data, k, num_iterations=100):
    np.random.seed(0)
    centroids = data.sample(n=k, random_state=0).values
    for _ in range(num_iterations):
        distances = np.linalg.norm(data.values[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == i].mean().values for i in range(k)])
        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids
    return labels

kmeans_labels = k_means(data_scaled, num_clusters)

# Manual K-Medoids clustering
def k_medoids(data, k, num_iterations=100):
    np.random.seed(0)
    medoids_indices = np.random.choice(data.shape[0], k, replace=False)
    medoids = data.values[medoids_indices]
    for _ in range(num_iterations):
        distances = np.linalg.norm(data.values[:, np.newaxis] - medoids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_medoids = np.array([data.values[labels == i].mean(axis=0) for i in range(k)])
        if np.all(new_medoids == medoids):
            break
        medoids = new_medoids
    return labels

kmedoids_labels = k_medoids(data_scaled, num_clusters)

# Visualize the clusters and count the number of points in each cluster
plt.figure(figsize=(12, 6))

# Select "radius_mean" and "texture_mean" for plotting
X_plot = data_imputed[['radius_mean', 'texture_mean']].values

# Plot K-Means clusters
plt.subplot(1, 2, 1)
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5)
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.title('Manual K-Means Clustering')
plt.colorbar(label='Cluster')

# Plot K-Medoids clusters
plt.subplot(1, 2, 2)
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=kmedoids_labels, cmap='viridis', alpha=0.5)
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.title('Manual K-Medoids Clustering')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.show()
