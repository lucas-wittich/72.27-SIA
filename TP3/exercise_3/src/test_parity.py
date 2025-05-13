import numpy as np
from perceptron import MultiLayerPerceptron
from optimizer import SGD_Optimizer

with open("../data/TP3-ej3-digitos.txt") as f:
    lines = [line.strip() for line in f if line.strip()]


blocks = [lines[i:i+7] for i in range(0, len(lines), 7)]
X = np.array([
    [int(bit) for row in block for bit in row.split()]
    for block in blocks
])

digits = np.arange(len(blocks))
y01 = (digits % 2).reshape(-1, 1)

mlp = MultiLayerPerceptron(
    layer_sizes=[35, 16, 1],
    optimizer=SGD_Optimizer(0.05),
    activations=['tanh', 'sigmoid'],
    learning_rate=0.05,
    batch_size=10,
    shuffle=False,
    loss_kind='cross_entropy',
    verbose=True
)

mlp.fit(X, y01, epochs=500)


probs = mlp.predict_proba_batch(X).ravel()       # shape (10,), float in [0,1]
preds = (probs >= 0.5).astype(int)               # threshold at 0.5
acc = (preds.reshape(-1, 1) == y01).mean()

print(f"\nParity detection accuracy: {acc*100:.1f}%\n")


print("Digit | Predicted | Actual")
print("---------------------------")
for d, p in zip(digits, preds):
    print(f"  {d:>2d}  |   {'odd' if p else 'even'}   |  {'odd' if (d % 2) else 'even'}")
