import numpy as np


class OjaNetwork:
    def __init__(self, dim: int, learning_rate: float, epochs: int):
        self.dim = dim
        self.lr = learning_rate
        self.epochs = epochs
        self.w = np.random.randn(dim)
        self.w /= np.linalg.norm(self.w)

    def train(self, X: np.ndarray):
        for _ in range(self.epochs):
            for x in X:
                y = np.dot(self.w, x)
                self.w += self.lr * (y * x - y**2 * self.w)
            self.w /= np.linalg.norm(self.w)

    def get_principal_component(self):
        return self.w
