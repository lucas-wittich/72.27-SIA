import numpy as np


class MultiLayerPerceptron():

    def __init__(self, layer_sizes, optimizer, activations=None, learning_rate=0.01, batch_size=None,
                 shuffle=True, loss_kind='cross_entropy', verbose=False):
        assert len(layer_sizes) >= 2
        if activations is None:
            activations = ['relu'] * (len(layer_sizes)-2) + ['softmax']
        assert len(activations) == len(layer_sizes)-1

        self.optimizer = optimizer
        self.dimensions = layer_sizes
        self.activations = activations
        self.lr = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loss_kind = loss_kind
        self.verbose = verbose

        # place to store metrics
        self.losses, self.accuracies = [], []

        # He‐normal initialization for all layers
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            std = np.sqrt(2.0 / fan_in)
            self.weights.append(np.random.randn(fan_out, fan_in) * std)
            self.biases.append(np.zeros((fan_out, 1)))

    def _activate(self, z, kind):
        if kind == 'relu':
            return np.maximum(0, z)
        if kind == 'sigmoid':
            return 1/(1+np.exp(-z))
        if kind == 'tanh':
            return np.tanh(z)
        if kind == 'softmax':
            exps = np.exp(z - z.max(axis=0, keepdims=True))
            return exps / exps.sum(axis=0, keepdims=True)
        raise ValueError(kind)

    def _activate_prime(self, z, kind):
        if kind == 'relu':
            return (z > 0).astype(float)
        if kind == 'sigmoid':
            s = 1/(1+np.exp(-z))
            return s*(1-s)
        if kind == 'tanh':
            return 1 - np.tanh(z)**2
        if kind == 'softmax':
            raise NotImplementedError
        raise ValueError(kind)

    def forward(self, X):
        a = X.T
        activations, zs = [a], []

        for W, b, act in zip(self.weights, self.biases, self.activations):
            z = W @ a + b
            zs.append(z)
            a = self._activate(z, act)
            activations.append(a)

        return activations, zs

    def backward(self, activations, zs, Y):
        m = Y.shape[1]

        grads_W = [np.zeros_like(W) for W in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        act_out = self.activations[-1]
        a_L, z_L = activations[-1], zs[-1]
        if act_out == 'softmax' and self.loss_kind == 'cross_entropy':
            delta = a_L - Y
        elif act_out == 'sigmoid' and self.loss_kind == 'cross_entropy':
            delta = a_L - Y
        else:
            delta = (a_L - Y) * self._activate_prime(z_L, act_out)

        grads_W[-1] = delta @ activations[-2].T / m
        grads_b[-1] = np.mean(delta, axis=1, keepdims=True)

        # hidden layers
        for l in range(2, len(self.dimensions)):
            z = zs[-l]
            act = self.activations[-l]
            sp = self._activate_prime(z, act)
            delta = (self.weights[-l+1].T @ delta) * sp
            grads_W[-l] = delta @ activations[-l-1].T / m
            grads_b[-l] = np.mean(delta, axis=1, keepdims=True)

        return grads_W, grads_b

    def compute_loss(self, y_pred, y_true, kind='mse'):
        if kind == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        else:
            p = np.clip(y_pred, 1e-16, 1 - 1e-16)
            return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def compute_accuracy(self, Y_pred, Y_true):
        preds = (Y_pred >= 0.5).astype(int)
        return np.mean(preds == Y_true)

    def update(self, grads_W, grads_b):
        params = self.weights + self.biases
        grads = grads_W + grads_b
        self.optimizer.step(params, grads)

    def train(self, X_batch, Y_batch):
        activations, zs = self.forward(X_batch)
        grads_W, grads_b = self.backward(activations, zs, Y_batch.T)
        self.update(grads_W, grads_b)

    def fit(self, X, Y, epochs=100):
        m = X.shape[0]
        bs = self.batch_size or m

        for epoch in range(1, epochs+1):
            if self.shuffle:
                perm = np.random.permutation(m)
                X, Y = X[perm], Y[perm]

            for start in range(0, m, bs):
                end = start + bs
                Xb = X[start:end]
                Yb = Y[start:end]
                self.train(Xb, Yb)

            y_pred = self.predict_proba_batch(X)  # shape (n, out)
            loss = self.compute_loss(y_pred, Y, kind=self.loss_kind)
            acc = self.compute_accuracy(y_pred, Y)
            self.losses.append(loss)
            self.accuracies.append(acc)

        if self.verbose:
            print(f"Epoch {epoch} — loss: {loss:.4f}, acc: {acc:.3f}")

    def predict_proba_batch(self, X):
        activations, _ = self.forward(X)
        return activations[-1].T

    def predict(self, X):
        probs = self.predict_proba_batch(X)
        if self.activations[-1] == 'softmax':
            return np.argmax(probs, axis=1)
        return (probs >= 0.5).astype(int)
