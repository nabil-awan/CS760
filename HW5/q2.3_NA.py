import pandas as pd
import numpy as np
from numpy.linalg import svd
from sklearn.preprocessing import StandardScaler

data2D = pd.read_csv('/data/data2D.csv', header=None).values
data1000D = pd.read_csv('/data/data1000D.csv', header=None).values

def buggy_pca(X, d):
    U, S, Vt = svd(X, full_matrices=False)
    Z = X @ Vt[:d].T
    reconstruction = Z @ Vt[:d]
    return Z, Vt[:d].T, reconstruction

def demeaned_pca(X, d):
    mean_X = np.mean(X, axis=0)
    X_centered = X - mean_X
    U, S, Vt = svd(X_centered, full_matrices=False)
    Z = X_centered @ Vt[:d].T
    reconstruction = Z @ Vt[:d] + mean_X
    return Z, Vt[:d].T, reconstruction

def normalized_pca(X, d):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    U, S, Vt = svd(X_scaled, full_matrices=False)
    Z = X_scaled @ Vt[:d].T
    reconstruction = Z @ Vt[:d] * scaler.scale_ + scaler.mean_
    return Z, Vt[:d].T, reconstruction

def dro(X, d):
    mean_X = np.mean(X, axis=0)
    X_centered = X - mean_X
    U, S, Vt = svd(X_centered, full_matrices=False)
    A = Vt[:d].T
    Z = X_centered @ A
    reconstruction = Z @ A.T + mean_X
    return Z, A, reconstruction

Z_buggy_2D, A_buggy_2D, recon_buggy_2D = buggy_pca(data2D, d=1)
Z_demeaned_2D, A_demeaned_2D, recon_demeaned_2D = demeaned_pca(data2D, d=1)
Z_normalized_2D, A_normalized_2D, recon_normalized_2D = normalized_pca(data2D, d=1)
Z_dro_2D, A_dro_2D, recon_dro_2D = dro(data2D, d=1)

reconstruction_errors_2D = {
    'Buggy PCA': np.sum((data2D - recon_buggy_2D) ** 2),
    'Demeaned PCA': np.sum((data2D - recon_demeaned_2D) ** 2),
    'Normalized PCA': np.sum((data2D - recon_normalized_2D) ** 2),
    'DRO': np.sum((data2D - recon_dro_2D) ** 2),
}

reconstruction_errors_2D

### Choose d ###
# To choose 'd' for the 1000D dataset, we look at the singular values from the SVD
# We are looking for a 'knee' in the plot of singular values which indicates a drop-off in the variance captured by the singular vectors

# Perform SVD on the 1000D data
U_1000D, S_1000D, Vt_1000D = svd(data1000D, full_matrices=False)

# Plot the singular values to find the 'knee'
plt.figure(figsize=(10, 6))
plt.plot(S_1000D, 'o-', markersize=4)
plt.yscale('log')  # Log scale to see the drop-off more clearly
plt.title('Singular Values of the 1000D Dataset')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.grid(True)


# The index of the 'knee' can be subjectively chosen by looking at the plot
# Usually, it's where the singular values start to flatten out significantly
# Here, we will implement a simple algorithmic approach to find the knee
# by calculating the angle between each point and its neighbors

# Calculate the angles between each point
angles = np.diff(np.log(S_1000D))
knee_index = np.argmin(angles) + 1  # +1 because diff reduces the array size by 1

# The chosen 'd' and justification will be based on the knee index
chosen_d = knee_index
justification = f"The 'knee' in the spectrum occurs at index {knee_index}, suggesting that 'd' should be chosen as {chosen_d}."

singular_values_plot_path, chosen_d, justification

##### Reconstruction Errors with d=31 ################

# First, let's define the functions for the different PCA methods and DRO:
def buggy_pca(X, d):
    # Perform SVD on the data matrix
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # Select the top-d components for the representation
    Z = U[:, :d] * s[:d]
    # Reconstruction
    X_reconstructed = Z @ Vt[:d, :]
    return Z, Vt[:d, :], X_reconstructed

def demeaned_pca(X, d):
    # Center the data
    mean_X = np.mean(X, axis=0)
    X_centered = X - mean_X
    # Perform SVD on the centered data
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Select the top-d components for the representation
    Z = U[:, :d] * s[:d]
    # Reconstruction
    X_reconstructed = Z @ Vt[:d, :] + mean_X
    return Z, Vt[:d, :], X_reconstructed, mean_X

def normalized_pca(X, d):
    # Center and scale the data
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    X_normalized = (X - mean_X) / std_X
    # Perform SVD on the normalized data
    U, s, Vt = np.linalg.svd(X_normalized, full_matrices=False)
    # Select the top-d components for the representation
    Z = U[:, :d] * s[:d]
    # Reconstruction
    X_reconstructed = (Z @ Vt[:d, :]) * std_X + mean_X
    return Z, Vt[:d, :], X_reconstructed, mean_X, std_X

def dro(X, d):
    # Center the data
    mean_X = np.mean(X, axis=0)
    X_centered = X - mean_X
    # Perform SVD on the centered data
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Select the top-d components for the representation
    A = Vt[:d, :].T
    Z = X_centered @ A
    # Reconstruction
    X_reconstructed = Z @ A.T + mean_X
    return Z, A, X_reconstructed, mean_X

# Now we will calculate the reconstruction errors for each method:
def reconstruction_error(original, reconstructed):
    return np.sum((original - reconstructed) ** 2)

# Buggy PCA
_, _, X_reconstructed_buggy = buggy_pca(data_1000D.values, d=31)
reconstruction_error_buggy = reconstruction_error(data_1000D.values, X_reconstructed_buggy)

# Demeaned PCA
_, _, X_reconstructed_demeaned, _ = demeaned_pca(data_1000D.values, d=31)
reconstruction_error_demeaned = reconstruction_error(data_1000D.values, X_reconstructed_demeaned)

# Normalized PCA
_, _, X_reconstructed_normalized, _, _ = normalized_pca(data_1000D.values, d=31)
reconstruction_error_normalized = reconstruction_error(data_1000D.values, X_reconstructed_normalized)

# DRO
_, _, X_reconstructed_dro, _ = dro(data_1000D.values, d=31)
reconstruction_error_dro = reconstruction_error(data_1000D.values, X_reconstructed_dro)

(reconstruction_error_buggy, reconstruction_error_demeaned, reconstruction_error_normalized, reconstruction_error_dro)
