import numpy  as np

np.random.seed(42)
X = np.random.rand(300,3) * 100
k=5
tol = 1e-3
max_iters = 100
n_samples, n_features = X.shape
centroids_init = X[np.random.choice(n_samples, k, replace=False)]


# vectorized version
centroids = centroids_init
for iteration in range(max_iters):
    dist = np.linalg.norm(X[:,np.newaxis,:]-centroids, axis=2)
    labels = np.argmin(dist,axis=1)
    # 300 x 5
    one_hot = np.eye(k)[labels]
    # 5x3
    new_center_sum = one_hot.T @ X
    # 5x1
    new_center_num = np.sum(one_hot.T, axis=1, keepdims=True)
    mask = new_center_num[:,0]>0
    new_center_sum = new_center_sum[mask]
    new_center_num = new_center_num[mask]
    new_centroids = new_center_sum/new_center_num

    if(new_centroids.shape[0]==centroids.shape[0] and
       np.all(np.linalg.norm(new_centroids-centroids,axis=1)<tol)):
        break
    centroids = new_centroids

print(centroids)

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
