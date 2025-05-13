import numpy as np
# Write a convolution operator
# Input: 
# 1. Data: A numpy array of size HxWxC1
# 2. Kernel: A numpy array of kxkxC1xC2

# Output: A numpy array which convolves the data with the provided kernel
# Assume, stride = 1, Padding is None. 
# Assume k is odd and is >= 3

def conv2d(X: np.array,kernel: np.array,stride:float):
    h,w,c = X.shape
    k1,k2,C1,C2= kernel.shape
    output_height = (h-k1)//stride+1
    output_width = (w-k2)//stride+1
    output_matrix = np.zeros((output_height,output_width,C2))
    
    for i in range(output_height):
        for j in range(output_width):
                for c2 in range(C2):
                    sub_x = X[i*stride : i*stride+k1,j*stride : j*stride+k2,:]
                    sub = np.sum(sub_x * kernel[:,:,:,c2])
                    output_matrix[i,j,c2] = sub
    return output_matrix

X = np.random.rand(5,5,3)
kernel = np.random.rand(1,1,3,4)
output = conv2d(X,kernel,1)
print(output.shape)