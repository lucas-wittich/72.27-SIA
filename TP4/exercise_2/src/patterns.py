import numpy as np
import matplotlib.pyplot as plt


def letter_templates():
    """
    Returns a dict of 4 patterns, each a 5×5 array of +1/−1.
    """
    patterns = {}
    patterns['J'] = np.array([
        [1,  1,  1,  1,  1],
        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1],
        [1, -1, -1,  1, -1],
        [1,  1,  1, -1, -1],
    ])

    patterns['A'] = np.array([
        [-1, -1, 1, -1, -1],
        [-1, 1, -1, 1, -1],
        [-1, 1, 1, 1, -1],
        [-1, 1, -1, 1, -1],
        [1, -1, -1, -1, 1]
    ])

    patterns['B'] = np.array([
        [1, 1, 1, 1, -1],
        [1, -1, -1, -1, 1],
        [1, 1, 1, 1, -1],
        [1, -1, -1, -1, 1],
        [1, 1, 1, 1, -1]
    ])

    patterns['P'] = np.array([
        [1, 1, 1, 1, -1],
        [1, -1, -1, -1, 1],
        [1, 1, 1, 1, -1],
        [1, -1, -1, -1, -1],
        [1, -1, -1, -1, -1]
    ])
    return patterns


def flatten(pattern: np.ndarray) -> np.ndarray:
    """5×5 → (25,) vector of +1/−1"""
    return pattern.reshape(-1)


def unflatten(vec: np.ndarray) -> np.ndarray:
    """(25,) → 5×5 array"""
    return vec.reshape(5, 5)


def add_noise(vec: np.ndarray, flip_fraction: float) -> np.ndarray:
    """Flip `flip_fraction` of bits in the ±1 vector."""
    noisy = vec.copy()
    n_flip = int(len(vec) * flip_fraction)
    idx = np.random.choice(len(vec), size=n_flip, replace=False)
    noisy[idx] *= -1
    return noisy


def plot_pattern(vec: np.ndarray,
                 ax: plt.Axes,
                 title: str = None):
    """
    Show a single 5×5 ±1‐pattern on ax.
    """
    arr = unflatten(vec)                # back to 5×5
    # imshow with grey for +1, white for –1
    ax.imshow(arr, cmap="gray_r", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def plot_trajectory(traj: list[np.ndarray],
                    suptitle: str = None):
    """
    Given the list of states (len T),
    plot them side by side.
    """
    T = len(traj)
    fig, axes = plt.subplots(1, T, figsize=(2*T, 2))
    for i, (state, ax) in enumerate(zip(traj, axes)):
        plot_pattern(state, ax, title=f"t={i}")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.show()
