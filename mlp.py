import numpy as np
from single_layer import SingleLayerFCN

class SimpleMLP:
    def __init__(self, input_size, output_size, hidden_size, lr):
        self.layer1 = SingleLayerFCN(input_size,hidden_size, lr)
        self.layer2 = SingleLayerFCN(hidden_size,output_size, lr)
    
    def relu(self,X):
        mask = (X>0).astype(float)
        #self.leaky_mask = (X >= 0).astype(float) + (X < 0).astype(float) * alpha
        self.mask = mask
        return X * mask
    
    def forward(self,X):
        y = self.layer1.forward(X)
        y = self.relu(y)
        y = self.layer2.forward(y)
        return y
    
    def compute_loss(self,y,y_true):
        return np.mean((y-y_true)**2, keepdims=False)
    
    def backward(self,y,y_true):
        m = y.shape[0]
        dout = (y-y_true) * (2/m)
        dout = self.layer2.backward_grad(dout)
        dout = dout * self.mask
        dout = self.layer1.backward_grad(dout)


def main():
    np.random.seed(42)

    input_size = 3
    hidden_size = 5
    output_size = 1
    lr = 0.05
    epochs = 100

    # Create toy regression dataset
    X = np.random.randn(200, input_size)
    true_W = np.array([[1.5], [-2.0], [0.7]])
    y = X @ true_W + 0.1 * np.random.randn(200, 1)

    # Initialize model
    model = SimpleMLP(input_size, hidden_size, output_size, lr)

    # Train
    for epoch in range(epochs):
        y_pred = model.forward(X)
        loss = model.compute_loss(y_pred, y)
        model.backward(y_pred,y)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    # Print example prediction
    print("\nTrue vs. Predicted (first 5):")
    print(np.hstack([y[:5], model.forward(X[:5])]))

if __name__ == "__main__":
    main()