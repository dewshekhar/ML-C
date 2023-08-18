import pandas as pd             # To work with data in tables
import numpy as np              # For math calculations
import matplotlib.pyplot as plt # To create plots

# Load the dataset from the provided link
url = "https://raw.githubusercontent.com/dewshekhar/DFS-and-BFS-for-8-Puzzel/main/cancer.csv"
data = pd.read_csv(url)

# Drop columns that are not numbers and those with all missing values
numeric_data = data.select_dtypes(include=[np.number])
numeric_data = numeric_data.loc[:, ~numeric_data.columns.duplicated()]
numeric_data = numeric_data.dropna(axis=1, how='all')

# Fill missing values with average values
imputed_data = numeric_data.fillna(numeric_data.mean())

# Standardize the data
scaled_data = (imputed_data - imputed_data.mean()) / imputed_data.std()

# Number of clusters we want
num_clusters = 2

# Custom K-Means clustering
def custom_k_means(data, k, num_iterations=100):
    np.random.seed(0)
    # Initialize random cluster centers
    centers = data.sample(n=k, random_state=0).values
    for _ in range(num_iterations):
        # Calculate distances from data points to centers
        distances = np.linalg.norm(data.values[:, np.newaxis] - centers, axis=2)
        # Assign each data point to the nearest center
        cluster_labels = np.argmin(distances, axis=1)
        # Calculate new centers by finding the average of points in each cluster
        new_centers = np.array([data[cluster_labels == i].mean().values for i in range(k)])
        # Check for convergence
        if np.all(new_centers == centers):
            break
        centers = new_centers
    return cluster_labels

kmeans_labels = custom_k_means(scaled_data, num_clusters)

# Custom K-Medoids clustering
def custom_k_medoids(data, k, num_iterations=100):
    np.random.seed(0)
    # Initialize random medoids indices
    medoids_indices = np.random.choice(data.shape[0], k, replace=False)
    medoids = data.values[medoids_indices]
    for _ in range(num_iterations):
        # Calculate distances between data points and medoids
        distances = np.linalg.norm(data.values[:, np.newaxis] - medoids, axis=2)
        # Assign data points to the nearest medoid
        cluster_labels = np.argmin(distances, axis=1)
        # Calculate new medoids as the data point with the smallest total distance to other points in its cluster
        new_medoids = np.array([data.values[cluster_labels == i].mean(axis=0) for i in range(k)])
        # Check for convergence
        if np.all(new_medoids == medoids):
            break
        medoids = new_medoids
    return cluster_labels

kmedoids_labels = custom_k_medoids(scaled_data, num_clusters)

# Plot the clusters and count points in each cluster
plt.figure(figsize=(12, 6))

# Choose two attributes for plotting
plot_data = imputed_data[['radius_mean', 'texture_mean']].values

# Plot K-Means clusters
plt.subplot(1, 2, 1)
plt.scatter(plot_data[:, 0], plot_data[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5)
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.title('Custom K-Means Clustering')
plt.colorbar(label='Cluster')

# Plot K-Medoids clusters
plt.subplot(1, 2, 2)
plt.scatter(plot_data[:, 0], plot_data[:, 1], c=kmedoids_labels, cmap='viridis', alpha=0.5)
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.title('Custom K-Medoids Clustering')
plt.colorbar(label='Cluster')

# Display author information
plt.figtext(0.5, 0.01, 'Student: DEWASHISH PRAMANIK\nAdmission No: IITP001202', ha='center', fontsize=10)

plt.tight_layout()
plt.show()
