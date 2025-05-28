import pandas as pd
import numpy as np


def load_europe_data(path: str = "../data/europe.csv"):
    df = pd.read_csv(path)
    countries = df.pop("Country")
    X = df.values.astype(float)
    X_std, means, stds = standardize(X)
    return X_std, countries


def standardize(X: np.ndarray):
    """
    Column-wise Z-score normalization.
    Returns:
      X_std : (n_samples, n_features) array of normalized values
      means : (n_features,) feature means
      stds  : (n_features,) feature standard deviations
    """
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    # avoid division by zero
    stds[stds == 0] = 1.0
    X_std = (X - means) / stds
    return X_std, means, stds
