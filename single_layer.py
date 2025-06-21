
import numpy as np
class SingleLayerFCN:
    def __init__(self, input_size, output_size, lr):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.weights = np.random.rand(input_size,output_size) # input_size x output_size
        self.bias = np.random.rand(1,output_size) # 1 x output_size
    
    def forward(self,X):
        self.X= X
        # N x input_size
        self.out = X@self.weights+self.bias
        return self.out
    
    def compute_loss(self,y,y_true):
        loss = np.mean((y_true - y)**2,keepdims=False)
        return loss

    def backward_grad(self, dout):
        # dout N x output_size
        num = self.X.shape[0]
        dw = self.X.T @ dout # input_size x output_size
        db = np.ones((1,num))@dout # 1 x output_size
        # db = np.sum(dout, axis=0, keepdims=True)
        self.weights -= dw * self.lr
        self.bias -= db * self.lr
        return dout @ self.weights.T
    
    def backward(self,y_true):
        # N x input_size
        num = self.X.shape[0]
        dout = (self.out-y_true) * (2/num) # N x output_size
        _ = self.backward_grad(dout)

# np.sqrt vs **2 !!!!!!!!!!!!!!!!!!!!!!!!!!

def main():
    np.random.seed(42)

    # Hyperparameters
    input_size = 3
    output_size = 1
    lr = 0.1
    epochs = 100

    # Create synthetic linear data
    X = np.random.randn(100, input_size)
    true_W = np.array([[2.0], [-1.0], [0.5]])
    y = X @ true_W + 0.1 * np.random.randn(100, 1)  # add small noise

    # Initialize model
    model = SingleLayerFCN(input_size, output_size, lr)

    # Training loop
    for epoch in range(epochs):
        y_pred = model.forward(X)
        loss = model.compute_loss(y_pred, y)
        model.backward(y)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    # Final weights vs true weights
    print("\nTrue Weights:\n", true_W.flatten())
    print("Learned Weights:\n", model.weights.flatten())

if __name__ == "__main__":
    main()