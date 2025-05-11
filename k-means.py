import numpy  as np

np.random.seed(42)
X = np.random.rand(300,3) * 100
k=5
n_samples, n_features = X.shape
centroids = X[np.random.choice(n_samples, k, replace=False)]
