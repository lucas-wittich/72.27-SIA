import numpy as np
from perceptron import MultiLayerPerceptron
from optimizer import SGD_Optimizer
import matplotlib.pyplot as plt


with open("../data/TP3-ej3-digitos.txt") as f:
    lines = [l.strip() for l in f if l.strip()]
blocks = [lines[i:i+7] for i in range(0, len(lines), 7)]
X = np.array([[int(b) for row in block for b in row.split()]
              for block in blocks])   # shape (10,35)

digits = np.arange(10)                # [0,1,…,9]
Y_int = digits.copy()                 # same shape (10,)
Y_oh = np.eye(10)[Y_int]             # shape (10,10), one-hot rows

mlp = MultiLayerPerceptron(
    layer_sizes=[35, 16, 10],            # two layers: hidden=16, output=10
    optimizer=SGD_Optimizer(0.05),
    activations=['tanh', 'softmax'],     # softmax at the end
    learning_rate=0.05,
    batch_size=10,                      # full-batch on 10 examples
    shuffle=True,
    loss_kind='cross_entropy',
    verbose=True
)

mlp.fit(X, Y_oh, epochs=1000)


preds = mlp.predict(X)

acc = (preds == Y_int).mean()
print(f"Digit‐recognition accuracy: {acc*100:.1f}%")


print("True → Pred")
for true, p in zip(Y_int, preds):
    print(f"   {true}   →   {p}")


def add_noise(X, flip_prob):
    """
    Flip each bit of X independently with probability flip_prob.
    X: np.ndarray of 0/1 shape (n_samples, n_features)
    """
    Xn = X.copy()
    mask = np.random.rand(*X.shape) < flip_prob
    Xn[mask] = 1 - Xn[mask]
    return Xn


noise_levels = np.linspace(0.0, 0.5, 11)

accuracies = []
for p in noise_levels:
    np.random.seed(0)
    X_noisy = add_noise(X, p)
    preds = mlp.predict(X_noisy)

    acc = (preds == Y_int).mean()
    accuracies.append(acc)


plt.figure(figsize=(6, 4))
plt.plot(noise_levels * 100, accuracies, 'o-')
plt.xlabel('Noise Level (% bits flipped)')
plt.ylabel('Digit‐recognition Accuracy')
plt.title('Robustness to Input Noise')
plt.ylim(0, 1.0)
plt.grid(True)
plt.show()
