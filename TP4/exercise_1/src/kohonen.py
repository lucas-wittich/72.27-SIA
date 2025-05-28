import numpy as np


class SelfOrganizingMap:
    def __init__(self, m: int, n: int, dim: int, learning_rate: float, sigma: float, epochs: int):
        self.m, self.n = m, n
        self.dim = dim
        self.learning_rate0 = learning_rate
        self.sigma0 = sigma
        self.epochs = epochs
        self.weights = np.random.randn(m * n, dim)

    def _neighbourhood(self, bmu_idx: int, epoch: int) -> np.ndarray:
        sigma = self.sigma0 * np.exp(-epoch / self.epochs)
        coords = np.array([[i, j] for i in range(self.m) for j in range(self.n)])
        bmu_coord = coords[bmu_idx]
        dist_sq = np.sum((coords - bmu_coord)**2, axis=1)
        return np.exp(-dist_sq / (2 * sigma**2))

    def _learning_rate(self, epoch: int) -> float:
        return self.learning_rate0 * np.exp(-epoch / self.epochs)

    def train(self, X: np.ndarray):
        for epoch in range(self.epochs):
            lr = self._learning_rate(epoch)
            for x in X:
                sq_dists = np.sum((self.weights - x)**2, axis=1)
                bmu_idx = np.argmin(sq_dists)

                h = self._neighbourhood(bmu_idx, epoch)[:, np.newaxis]  # shape (n_neurons, 1)
                self.weights += lr * h * (x - self.weights)

    def map_data(self, X: np.ndarray) -> np.ndarray:
        sq_dists = ((X[:, np.newaxis, :] - self.weights[np.newaxis, :, :])**2).sum(axis=2)
        return np.argmin(sq_dists, axis=1)

    def umatrix(self) -> np.ndarray:
        coords = [(i, j) for i in range(self.m) for j in range(self.n)]
        um = np.zeros((self.m, self.n))
        for idx, (i, j) in enumerate(coords):
            neighbor_idxs = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < self.m and 0 <= nj < self.n:
                    neighbor_idxs.append(ni*self.n + nj)
            um[i, j] = np.mean(
                np.linalg.norm(self.weights[idx] - self.weights[neighbor_idxs], axis=1)
            )
        return um
