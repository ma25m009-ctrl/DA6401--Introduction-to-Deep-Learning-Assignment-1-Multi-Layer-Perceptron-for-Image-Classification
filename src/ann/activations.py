import numpy as np


class Sigmoid:
    def forward(self, x):
        # Numerically stable: avoids overflow for large negative x
        self.out = np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)


class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1 - self.out ** 2)


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Softmax:
    def forward(self, x):
        # Subtract row-wise max for numerical stability
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.out

    def backward(self, grad):
        # Combined Softmax + CrossEntropy gradient = (y_pred - y_true)
        # already handled in CrossEntropy.backward() — pass through unchanged
        return grad
