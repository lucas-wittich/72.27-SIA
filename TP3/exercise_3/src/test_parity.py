import numpy as np
from perceptron import MultiLayerPerceptron
from optimizer import SGD_Optimizer

# 1) Load and parse the 7×5 digit file
with open("../data/TP3-ej3-digitos.txt") as f:
    # strip out any empty lines
    lines = [line.strip() for line in f if line.strip()]


blocks = [lines[i:i+7] for i in range(0, len(lines), 7)]


X = np.array([
    [int(bit) for row in block for bit in row.split()]
    for block in blocks
])   # shape (10, 35)

# 3) Build parity labels: even digit → 0, odd digit → 1
digits = np.arange(len(blocks))           # [0,1,2,…,9]
y01 = (digits % 2).reshape(-1, 1)       # shape (10,1), values in {0,1}

# 4) Instantiate the MLP
mlp = MultiLayerPerceptron(
    layer_sizes=[35, 16, 1],           # 35 inputs → 16 hidden → 1 output
    optimizer=SGD_Optimizer(0.05),     # single SGD optimizer
    activations=['tanh', 'sigmoid'],    # tanh hidden, sigmoid output
    learning_rate=0.05,
    batch_size=10,                     # full‐batch on 10 examples
    shuffle=False,
    loss_kind='cross_entropy',
    verbose=True
)

# 5) Train for 500 epochs
mlp.fit(X, y01, epochs=500)

# 6) Predict and compute accuracy
probs = mlp.predict_proba_batch(X).ravel()       # shape (10,), float in [0,1]
preds = (probs >= 0.5).astype(int)               # threshold at 0.5
acc = (preds.reshape(-1, 1) == y01).mean()

print(f"\nParity detection accuracy: {acc*100:.1f}%\n")

# 7) Show per‐digit results
print("Digit | Predicted | Actual")
print("---------------------------")
for d, p in zip(digits, preds):
    print(f"  {d:>2d}  |   {'odd' if p else 'even'}   |  {'odd' if (d % 2) else 'even'}")
