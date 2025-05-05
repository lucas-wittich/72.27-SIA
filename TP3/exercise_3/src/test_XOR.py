import numpy as np
from perceptron import MultiLayerPerceptron
from optimizer import SGD_Optimizer

X = np.array([
    [-1,  1],
    [1, -1],
    [-1, -1],
    [1,  1],
])
y_sign = np.array([1,  1, -1, -1])

y = ((y_sign + 1) / 2).reshape(-1, 1)

mlp_xor = MultiLayerPerceptron(
    layer_sizes=[2, 2, 1],
    optimizer=SGD_Optimizer(0.1),
    activations=['tanh', 'sigmoid'],
    learning_rate=0.1,
    batch_size=4,
    shuffle=False,
    loss_kind='cross_entropy',
    verbose=True
)

mlp_xor.fit(X, y, epochs=1000)


probs = mlp_xor.predict_proba_batch(X)
preds = mlp_xor.predict(X)

print("Raw sigmoid outputs:     ", np.round(probs.flatten(), 3))
print("Thresholded predictions:", preds.flatten())
print("True XOR labels (0/1):  ", y.flatten())
