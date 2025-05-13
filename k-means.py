import numpy  as np

np.random.seed(42)
X = np.random.rand(300,3) * 100
k=5
tol = 1e-3
max_iters = 100
n_samples, n_features = X.shape
centroids_init = X[np.random.choice(n_samples, k, replace=False)]


# for loop version
centroids = centroids_init
for iteration in range(max_iters):
    dist = np.linalg.norm(X[:,np.newaxis]-centroids,axis=2) # 300x5
    labels = np.argmin(dist,axis=1) #300x1
    new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(k)])

    # Check for convergence
    if centroids.shape[0]==new_centroids.shape[0] and \
    np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
        break
    centroids = new_centroids

print(centroids)

# vectorized version
centroids = centroids_init
for iteration in range(max_iters):
    dist = np.linalg.norm(X[:,np.newaxis]-centroids,axis=2) # 300x5
    labels = np.argmin(dist,axis=1) #300x1
    # 5x300 @ 300x3 -> 5x3
    one_hot = np.eye(k)[labels] #300x5
    sum_per_center = one_hot.T @ X #5X3
    num_per_center = one_hot.T.sum(axis=1,keepdims=True) #5x1
    mask = num_per_center[:,-1]>0
    sum_per_center = sum_per_center[mask]
    num_per_center = num_per_center[mask]
    new_centroids = sum_per_center / num_per_center

    # Check for convergence
    if centroids.shape[0]==new_centroids.shape[0] and \
    np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
        break
    centroids = new_centroids

print(centroids)
