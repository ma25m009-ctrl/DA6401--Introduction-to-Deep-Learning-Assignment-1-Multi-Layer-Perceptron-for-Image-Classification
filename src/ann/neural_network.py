import numpy as np


class NeuralNetwork:

    def __init__(self, layers, activations, loss_fn, optimizer, weight_decay=0.0):
        self.layers       = layers
        self.activations  = activations
        self.loss_fn      = loss_fn
        self.optimizer    = optimizer
        self.weight_decay = weight_decay

    def forward(self, X):
        for layer, activation in zip(self.layers, self.activations):
            X = layer.forward(X)
            X = activation.forward(X)
        return X

    def backward(self, y_true, y_pred):
        grad = self.loss_fn.backward()
        for layer, activation in reversed(list(zip(self.layers, self.activations))):
            grad = activation.backward(grad)
            grad = layer.backward(grad, self.weight_decay)

    def update_weights(self):
        for layer in self.layers:
            self.optimizer.update(layer)

    def train(self, X, y, epochs, batch_size):
        for epoch in range(epochs):

            # Shuffle without mutating the original arrays
            perm = np.random.permutation(len(X))
            X_s  = X[perm]
            y_s  = y[perm]

            epoch_loss  = 0.0
            num_batches = 0

            for i in range(0, len(X_s), batch_size):
                xb = X_s[i:i + batch_size]
                yb = y_s[i:i + batch_size]

                y_pred = self.forward(xb)
                loss   = self.loss_fn.forward(y_pred, yb)

                epoch_loss  += loss
                num_batches += 1

                self.backward(yb, y_pred)
                self.update_weights()

        return epoch_loss / num_batches

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
