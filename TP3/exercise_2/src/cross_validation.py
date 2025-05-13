import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from perceptron import LinearPerceptron, NonLinearPerceptron


def k_fold_scores(ModelClass, X, y, k=5, num_epochs=100, random_state=42, **model_kwargs):
    n = len(X)
    indices = list(range(n))
    random.seed(random_state)
    np.random.seed(random_state)
    random.shuffle(indices)

    fold_size = n // k
    scores = []

    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        model = ModelClass(num_inputs=X.shape[1], **model_kwargs)
        model.fit(X_train, y_train, num_epochs=num_epochs)

        scores.append(model.evaluate(X_val, y_val))

    return sum(scores) / k, scores


if __name__ == "__main__":
    df = pd.read_csv("../data/TP3-ej2-conjunto.csv")
    threshold = df['y'].median()
    df['label01'] = (df['y'] > threshold).astype(int)

    X = df[['x1', 'x2', 'x3']].values
    y01 = df['label01'].values
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # ——— Linear perceptron (±1 labels) ———
    y_pm = np.where(y01 == 1,  1, -1)
    mean_lin, scores_lin = k_fold_scores(
        LinearPerceptron,
        X_scaled, y_pm,
        k=5,
        learning_rate=0.1,
        num_epochs=100
    )
    print(f"Linear ±1 → CV mean acc: {mean_lin:.3f}, folds: {np.round(scores_lin, 3)}")

    # # ——— Nonlinear perceptron (sigmoid, 0/1 labels) ———
    # mean_nl, scores_nl = k_fold_scores(
    #     NonLinearPerceptron,
    #     X_scaled, y01,
    #     k=5,
    #     learning_rate=0.1,
    #     activation='sigmoid',
    #     num_epochs=100
    # )
    # print(f"NonLinear sigmoid → CV mean acc: {mean_nl:.3f}, folds: {np.round(scores_nl, 3)}")

    # # ——— Nonlinear perceptron (tanh, 0/1 labels) ———
    # mean_nl_tanh, scores_nl_tanh = k_fold_scores(
    #     NonLinearPerceptron,
    #     X_scaled, y01,
    #     k=5,
    #     learning_rate=0.1,
    #     activation='tanh',
    #     num_epochs=100
    # )
    # print(f"NonLinear tanh → CV mean acc: {mean_nl_tanh:.3f}, folds: {np.round(scores_nl_tanh, 3)}")
