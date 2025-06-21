import numpy as np
from single_layer import SingleLayerFCN

class MLPClassifier:
    def __init__(self, input_size, hidden_size, output_size, lr, softmax=True):
        self.layer1 = SingleLayerFCN(input_size, hidden_size,lr)
        self.layer2= SingleLayerFCN(hidden_size, output_size,lr)
        self.class_num = output_size
        self.softmax = softmax

    def relu(self,X):
        mask = (X>0).astype(float)
        self.mask = mask
        return X * mask
    
    def softmax(self,y):
       exp_y = np.exp(y-np.max(y,axis=1,keepdims=True))
       return exp_y / np.sum(exp_y,axis=1,keepdims=True)
    
    def sigmoid(self,y):
       return 1 / (1 + np.exp(-y))
    
    def forward(self, X):
        y = self.layer1.forward(X)
        y = self.relu(y)
        y = self.layer2.forward(y)
        if self.softmax:
            y = self.softmax(y)
        else:
            y = self.sigmoid(y)
        self.pred = y
        return y
    
    def compute_loss(self,probs,y_true):
        label = np.eye(self.class_num)[y_true]
        loss = -np.log(np.sum(probs * label, axis=1, keepdims=False))
        return np.mean(loss)
    
    def compute_loss_binary(self,probs,y_true):
        label = np.eye(self.class_num)[y_true]
        loss = -np.mean(label * np.log(probs) + (1-label) * np.log(1-probs))
        return loss

    # backward function is same for softmax and sigmoid
    def backward(self, y_true):
        m = y_true.shape[0]
        label = np.eye(self.class_num)[y_true]
        dout = (self.pred-label)/m
        # back propagate
        dout = self.layer2.backward_grad(dout)
        dout = self.mask * dout
        _ = self.layer1.backward_grad(dout)



def main():
    np.random.seed(42)
    N, D, H, C = 200, 5, 10, 3  # samples, input_dim, hidden, classes
    X = np.random.randn(N, D)
    y = np.random.randint(0, C, size=N)
    
    #model = MLPClassifier(input_size=D, hidden_size=H, output_size=C, lr=0.1, softmax=True)
    model = MLPClassifier(input_size=D, hidden_size=H, output_size=C, lr=0.1, softmax=False)

    for epoch in range(100):
        probs = model.forward(X)
        # softmax loss
        # loss = model.compute_loss(probs, y)
        # sigmoid loss
        loss = model.compute_loss_binary(probs, y)
        model.backward(y)
        if epoch % 10 == 0 or epoch == 99:
            preds = np.argmax(probs, axis=1)
            acc = np.mean(preds == y)
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.2f}")

if __name__ == "__main__":
    main()