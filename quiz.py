import numpy as np
# 🧱 Data Manipulation Questions (NumPy-focused)
# 🔹 Basic
# What is the difference between reshape(), ravel(), and flatten() in NumPy?
# How does broadcasting work in NumPy? Give an example.
# What’s the difference between np.stack and np.concatenate?
# How do you select the last column of a 2D array?
# How would you reverse the rows of a 2D array?
X = np.random.rand(3,3)
print(X)
print(X[:,::-1])

# 🔹 Intermediate
# Given an array of shape (100, 3), how would you normalize each row?
# How can you remove rows from a 2D array where the value in the second column is negative?
X = np.random.rand(10,3)-0.5
mask = X[:,1]>=0
X = X[mask]
print(X)
# How do you find the index of the maximum value in each row of a 2D array?
X = np.random.rand(10,3)-0.5
index = np.argmax(X,axis=1)
print(index)
# How would you compute pairwise distances between all rows in an array of shape (N, D)?
X = np.random.rand(10,3)
Y = np.random.rand(5,3)
dist = np.linalg.norm(X[:,np.newaxis] - Y,axis=2, keepdims=True)
print(dist.shape)
# What is the fastest way to apply a function across axis 1 of a 2D array?

# 🧮 Matrix Calculation Questions
# 🔹 Basic
# What’s the difference between np.dot, np.matmul, and the @ operator?
# What is the shape of the result when multiplying a (3, 4) matrix by a (4, 1) matrix?
# How do you compute the inverse or pseudo-inverse of a matrix in NumPy?
# What happens when you try to multiply two matrices with incompatible shapes?

# 🔹 Intermediate
# How can you represent a rigid-body transformation as a 4×4 matrix?
# How would you compute the determinant of a matrix, and what does it mean?
# How do you solve a linear system Ax=b using matrix operations?
# What is the effect of multiplying a vector by a rotation matrix versus a scaling matrix?

# 🔹 Advanced / Application
# Given 3D points and a transformation matrix, how do you project them onto a 2D image plane?
# How would you implement Principal Component Analysis (PCA) from scratch using only matrix ops?
# What is Singular Value Decomposition (SVD) used for in practical applications?

# 🎯 Challenge-Style Exercises
# Matrix Transformation:
# Given an array of 3D points, rotate them 90° around the Z-axis and then translate them by (1, 2, 3). Show the result.
# Flatten and Reshape:
# Flatten a (3, 4, 5) array to 2D, then reshape it back to the original shape.
X = np.random.rand(3,4,5)
X = X.reshape(3,20)
X = X.reshape(3,4,5)
# Batch Matrix Multiply:
# Given arrays A of shape (10, 3, 3) and B of shape (10, 3, 1), compute A[i] @ B[i] for all i.
X = np.random.rand(10,3,3)
Y = np.random.rand(10,3,1)
M = X@Y
print(M.shape)
# Normalization:
# Normalize a (100, 3) array so each point lies on the unit sphere.
X = np.random.rand(100,3)
norm = X/np.linalg.norm(X,axis=1,keepdims=True)
print(norm.shape)