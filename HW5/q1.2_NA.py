import numpy as np

# Set the seed for reproducibility
np.random.seed(0)

# Number of samples from each distribution
n_samples = 100

# Sigma parameter
sigma = 8

# Means and covariance matrices for the distributions
means = [np.array([-1, -1]), np.array([1, -1]), np.array([0, 1])]
covariances = [
    sigma * np.array([[2, 0.5], [0.5, 1]]),
    sigma * np.array([[1, -0.5], [-0.5, 2]]),
    sigma * np.array([[1, 0], [0, 2]])
]

# Generate samples
samples_a = np.random.multivariate_normal(means[0], covariances[0], n_samples)
samples_b = np.random.multivariate_normal(means[1], covariances[1], n_samples)
samples_c = np.random.multivariate_normal(means[2], covariances[2], n_samples)

# Combine the samples and the true labels
X = np.vstack((samples_a, samples_b, samples_c))
true_labels = np.array([0]*n_samples + [1]*n_samples + [2]*n_samples)

X[:10], true_labels[:10]  # Show the first 10 samples and their true labels
def initialize_centroids(X, k):
    """Randomly initialize centroids from the dataset X."""
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    """Assign each data point to the closest centroid."""
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(X, labels, k):
    """Calculate new centroids from the means of the points assigned to each centroid."""
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def converged(centroids, new_centroids):
    """Check if the centroids have stabilized."""
    return np.all(centroids == new_centroids)

def k_means_clustering(X, k, max_iters=100):
    """Implement K-means clustering."""
    # Initialize centroids
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        # Assign clusters
        labels = assign_clusters(X, centroids)
        # Update centroids
        new_centroids = update_centroids(X, labels, k)
        # Check for convergence
        if converged(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Apply K-means clustering on our dataset
k = 3  # We know we have 3 clusters
centroids, kmeans_labels = k_means_clustering(X, k)

centroids, kmeans_labels[:10]  # Show the final centroids and the first 10 labels
from scipy.stats import multivariate_normal

def initialize_gmm(X, k):
    """Initialize the GMM parameters."""
    n_samples, n_features = X.shape
    # Randomly initialize the means by choosing k data points
    means = initialize_centroids(X, k)
    # Initialize the covariances to identity matrices
    covariances = [np.eye(n_features) for _ in range(k)]
    # Initialize the mixing coefficients to uniform distribution
    mixing_coeffs = np.full(k, 1/k)
    return means, covariances, mixing_coeffs

def e_step(X, means, covariances, mixing_coeffs, k):
    """E-step: Compute the responsibilities."""
    responsibilities = np.zeros((X.shape[0], k))
    for i in range(k):
        responsibilities[:, i] = mixing_coeffs[i] * multivariate_normal.pdf(X, mean=means[i], cov=covariances[i])
    # Normalize responsibilities
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

def m_step(X, responsibilities):
    """M-step: Update the parameters."""
    n_samples, n_features = X.shape
    k = responsibilities.shape[1]
    means = np.zeros((k, n_features))
    covariances = [np.zeros((n_features, n_features)) for _ in range(k)]
    mixing_coeffs = np.zeros(k)

    for i in range(k):
        # Update means
        weights = responsibilities[:, i]
        total_weight = weights.sum()
        means[i] = (X * weights[:, np.newaxis]).sum(axis=0) / total_weight
        # Update covariances
        covariances[i] = np.cov(X.T, aweights=weights, ddof=0)
        # Update mixing coefficients
        mixing_coeffs[i] = total_weight / n_samples

    return means, covariances, mixing_coeffs

def gmm_log_likelihood(X, means, covariances, mixing_coeffs, k):
    """Calculate the log-likelihood of the data under the current GMM parameters."""
    log_likelihood = 0
    for i in range(k):
        log_likelihood += mixing_coeffs[i] * multivariate_normal.pdf(X, mean=means[i], cov=covariances[i])
    return np.log(log_likelihood).sum()

def gmm_em(X, k, max_iters=100, tol=1e-3):
    """EM algorithm for GMM."""
    # Initialize the parameters
    means, covariances, mixing_coeffs = initialize_gmm(X, k)
    log_likelihood = 0

    for iteration in range(max_iters):
        # E-step
        responsibilities = e_step(X, means, covariances, mixing_coeffs, k)
        # M-step
        means, covariances, mixing_coeffs = m_step(X, responsibilities)
        # Compute the log-likelihood
        new_log_likelihood = gmm_log_likelihood(X, means, covariances, mixing_coeffs, k)
        # Check for convergence
        if abs(new_log_likelihood - log_likelihood) < tol:
            break
        log_likelihood = new_log_likelihood

    return means, covariances, mixing_coeffs, responsibilities

# Apply EM for GMM on our dataset
means, covariances, mixing_coeffs, responsibilities = gmm_em(X, k)

means, mixing_coeffs, responsibilities[:10]  # Show the final means, mixing coefficients and the first 10 responsibilities

def k_means_objective(X, centroids, labels):
    """Calculate the K-means clustering objective."""
    return sum(np.linalg.norm(X[labels == i] - centroid, axis=1).sum() for i, centroid in enumerate(centroids))

# Calculate K-means clustering objective
kmeans_objective = k_means_objective(X, centroids, kmeans_labels)

# Since we have true labels, we can map each found cluster to the true labels.
# For each cluster label, we find the true label that is most common within that cluster.
# This will not always give a perfect mapping, it assumes a one-to-one relationship, which may not hold in all cases.
def map_labels(found_labels, true_labels):
    from scipy.stats import mode

    labels = np.zeros_like(found_labels)
    for i in range(k):
        mask = (found_labels == i)
        # Find the most common true label in each cluster
        labels[mask] = mode(true_labels[mask])[0]
    return labels

# Map the cluster labels to the true labels
mapped_kmeans_labels = map_labels(kmeans_labels, true_labels)
mapped_gmm_labels = map_labels(np.argmax(responsibilities, axis=1), true_labels)

# Calculate the accuracy of K-means and GMM
kmeans_accuracy = np.mean(mapped_kmeans_labels == true_labels)
gmm_accuracy = np.mean(mapped_gmm_labels == true_labels)

kmeans_objective, kmeans_accuracy, gmm_accuracy
gmm_clusters = np.argmax(responsibilities, axis=1)

# Calculate GMM clustering objective using the K-means objective function
gmm_objective = k_means_objective(X, means, gmm_clusters)

gmm_objective
