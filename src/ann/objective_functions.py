import numpy as np


class CrossEntropy:

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        # Clip to avoid log(0) and log(value > 1)
        y_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)
        loss = -np.mean(np.sum(y_true * np.log(y_clipped), axis=1))
        return loss

    def backward(self):
        # Combined gradient of Softmax + CrossEntropy = (y_pred - y_true) / batch
        return (self.y_pred - self.y_true) / self.y_true.shape[0]


class MSE:

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self):
        # np.mean divides by (batch * classes); gradient denominator must match
        return 2 * (self.y_pred - self.y_true) / (self.y_true.shape[0] * self.y_true.shape[1])
