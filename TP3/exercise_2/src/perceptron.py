from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split


class LinearPerceptron():

    def __init__(self, num_inputs, learning_rate):
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = learning_rate
        self.losses_mse = []
        self.losses_bce = []
        self.accuracies = []

    def linear(self, inputs):
        return inputs @ self.weights[1:].T + self.weights[0]

    def step_function(self, z):
        return 1 if z >= 0 else -1

    def predict_proba(self, inputs):
        return self.linear(inputs)

    def predict(self, inputs):
        z = self.linear(inputs)
        return self.step_function(z)

    def error(self, prediction, target):
        return target - prediction

    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = self.loss(prediction, target)
        self.weights[1:] += self.learning_rate * error * inputs
        self.weights[0] += self.learning_rate * error

    def fit(self, X, y, num_epochs, use_bce=False):
        self.errors_per_epoch = []

        for epoch in range(num_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                error = self.error(self.predict(xi), yi)
                if error != 0:
                    errors += 1
                    self.train(xi, yi)

            self.losses_mse.append(self.compute_MSE(X, y))
            self.losses_bce.append(self.compute_log_loss(X, y))
            self.accuracies.append(self.evaluate(X, y))
            self.errors_per_epoch.append(errors)

    def evaluate(self, X, y):
        preds = [self.predict(xi) for xi in X]
        return sum(p == t for p, t in zip(preds, y)) / len(y)

    def loss(self, prediction, target):
        return target - prediction

    def misclassifications(self, X, y):
        correct = sum(self.predict(xi) == yi for xi, yi in zip(X, y))
        return len(y) - correct

    def compute_MSE(self, X, y):
        preds = np.array([self.predict(xi) for xi in X])
        return np.mean((y - preds) ** 2)

    def compute_log_loss(self, X, y, epsilon=1e-16):
        zs = np.array([self.linear(xi) for xi in X])
        ps = 1 / (1 + np.exp(-zs))
        ps = np.clip(ps, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(ps) + (1 - y) * np.log(1 - ps))


class NonLinearPerceptron():

    def __init__(self, num_inputs, learning_rate=0.1, activation='sigmoid'):
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = learning_rate
        assert activation in ('sigmoid', 'tanh')
        self.activation = activation
        self.losses_mse = []
        self.losses_bce = []
        self.accuracies = []

    def linear(self, inputs):
        return inputs @ self.weights[1:].T + self.weights[0]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def predict(self, inputs):
        p = self.predict_proba(inputs)
        if self.activation == 'sigmoid':
            return 1 if p >= 0.5 else 0
        else:
            return 1 if p >= 0.0 else 0

    def predict_proba(self, inputs):
        z = self.linear(inputs)
        if self.activation == 'sigmoid':
            return self.sigmoid(z)
        else:
            return self.tanh(z)

    def evaluate(self, X, y):
        preds = [1 if self.predict(x) >= 0.5 else 0 for x in X]
        return sum(p == t for p, t in zip(preds, y)) / len(y)

    def loss(self, prediction, target, MSE=True):
        if MSE:
            return self.compute_MSE(prediction, target)
        else:
            return self.compute_log_loss(prediction, target)

    def compute_MSE(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2)

    def compute_log_loss(self, y_pred, y_true, epsilon=1e-16):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def train(self, inputs, target):
        z = self.linear(inputs)
        if self.activation == 'sigmoid':
            y_pred = self.sigmoid(z)
            deriv = self.sigmoid_derivative(z)
        else:
            y_pred = self.tanh(z)
            deriv = self.tanh_derivative(z)

        error = y_pred - target
        grad = error * deriv

        self.weights[1:] -= self.learning_rate * grad * inputs
        self.weights[0] -= self.learning_rate * grad

    def fit(self, X, y, num_epochs=100, use_bce=False):
        self.losses = []
        self.accuracies = []

        for epoch in range(num_epochs):
            for xi, yi in zip(X, y):
                self.train(xi, yi)

            probs = np.array([self.predict_proba(xi) for xi in X])
            if use_bce:
                epoch_loss = self.compute_log_loss(probs, y)
            else:
                epoch_loss = self.compute_MSE(probs, y)
            self.losses.append(epoch_loss)

            acc = self.evaluate(X, y)
            self.accuracies.append(acc)


if __name__ == "__main__":
    df = pd.read_csv("../data/TP3-ej2-conjunto.csv")

    threshold = df['y'].median()
    df['label01'] = (df['y'] > threshold).astype(int)

    X = df[['x1', 'x2', 'x3']].values
    y01 = df['label01'].values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Split into train/test sets (stratified)
    X_train, X_test, y_train01, y_test01 = train_test_split(
        X_scaled, y01, test_size=0.3, random_state=42, stratify=y01
    )
    # Prepare ±1 labels for linear perceptron
    y_train_pm = np.where(y_train01 == 1, 1, -1)
    y_test_pm = np.where(y_test01 == 1, 1, -1)

    lp = LinearPerceptron(num_inputs=X_train.shape[1], learning_rate=0.1)
    nlp_sig = NonLinearPerceptron(num_inputs=X_train.shape[1], learning_rate=0.1, activation='sigmoid')
    nlp_tanh = NonLinearPerceptron(num_inputs=X_train.shape[1], learning_rate=0.1, activation='tanh')

    num_epochs = 50
    lin_losses, lin_accs = [], []
    sig_losses, sig_accs = [], []
    tanh_losses, tanh_accs = [], []

    for epoch in range(num_epochs):
        # Linear epoch
        for xi, yi in zip(X_train, y_train_pm):
            lp.train(xi, yi)
        lin_preds = np.array([lp.predict(xi) for xi in X_train])
        lin_loss = np.mean((lin_preds - y_train_pm)**2)
        lin_acc = lp.evaluate(X_train, y_train_pm)
        lin_losses.append(lin_loss)
        lin_accs.append(lin_acc)

        # Sigmoid epoch
        for xi, yi in zip(X_train, y_train01):
            nlp_sig.train(xi, yi)
        probs_sig = np.array([nlp_sig.predict_proba(xi) for xi in X_train])
        sig_loss = nlp_sig.compute_MSE(probs_sig, y_train01)
        sig_acc = nlp_sig.evaluate(X_train, y_train01)
        sig_losses.append(sig_loss)
        sig_accs.append(sig_acc)

        # Tanh epoch
        for xi, yi in zip(X_train, y_train01):
            nlp_tanh.train(xi, yi)
        probs_tanh = np.array([nlp_tanh.predict_proba(xi) for xi in X_train])
        tanh_loss = nlp_tanh.compute_MSE(probs_tanh, y_train01)
        tanh_acc = nlp_tanh.evaluate(X_train, y_train01)
        tanh_losses.append(tanh_loss)
        tanh_accs.append(tanh_acc)

    print("Epoch | Lin Miscls | Lin Acc   | Sig Loss  | Sig Acc | Tanh Loss | Tanh Acc")
    print("-------------------------------------------------------------------------")
    for i in range(num_epochs):
        print(f"{i+1:5d} | "
              f"{lin_losses[i]:10d} | {lin_accs[i]:.4f} | "
              f"{sig_losses[i]:.4f} | {sig_accs[i]:.4f} | "
              f"{tanh_losses[i]:.4f} | {tanh_accs[i]:.4f}")
