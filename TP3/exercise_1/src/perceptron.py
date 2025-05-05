import numpy as np
from tqdm import tqdm


class Perceptron:

    def __init__(self, num_inputs, learning_rate):
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = learning_rate

    def linear(self, inputs):
        Z = inputs @ self.weights[1:].T + self.weights[0]
        return Z

    def step_function(self, t: int):
        if t >= 0:
            return 1
        else:
            return -1

    def predict(self, inputs):
        z = self.linear(inputs)
        return self.step_function(z)

    def loss(self, prediction, target):
        return target - prediction

    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = self.loss(prediction, target)
        self.weights[1:] += self.learning_rate * error * inputs
        self.weights[0] += self.learning_rate * error

    def fit(self, X, y, num_epochs):
        for epoch in range(num_epochs):
            for input, target in zip(X, y):
                self.train(np.array(input), target)


def train_AND():

    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    p = Perceptron(num_inputs=2, learning_rate=0.1)
    p.fit(X, y, num_epochs=10)

    for x in X:
        print(f"Input: {x}, Prediction: {p.predict(x)}")


def train_XOR():
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    p = Perceptron(num_inputs=2, learning_rate=0.1)
    p.fit(X, y, num_epochs=10)

    for x in X:
        print(f"Input: {x}, Prediction: {p.predict(x)}")


if __name__ == "__main__":
    train_AND()
    train_XOR()
