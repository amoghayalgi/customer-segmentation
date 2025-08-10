import numpy as np

class PCA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis = 0)
        X_centered = X - self.mean
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        #Sorts eigenvalues in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        #Get only eigenvectors for n components
        self.components = eigenvectors[:, sorted_idx[:self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean
        #Returns projection of X_centered on new PCA axes
        return np.dot(X_centered, self.components)