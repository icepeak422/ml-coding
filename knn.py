import numpy  as np

# X train set with 30 points and label
# test set 10 points
np.random.seed(42)
X = np.random.rand(30,3) * 100
labels = np.random.randint(0, 5, (30,1))
test = np.random.rand(10,3) * 100
k=5

# KNN start
dist = np.linalg.norm((test[:,np.newaxis,:] - X),axis=2,keepdims=True)
top_k_neighbor = np.argsort(dist,axis=1)[:,:k,:].sum(axis=2)
top_k_pred = labels[top_k_neighbor]
print(top_k_pred.shape)
def majority_vote(row):
    return np.bincount(row).argmax()

y_pred = np.apply_along_axis(majority_vote, axis=1, arr=top_k_pred)
print(y_pred)




