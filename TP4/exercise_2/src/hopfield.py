from matplotlib import animation, pyplot as plt
import numpy as np


class HopfieldNetwork:
    def __init__(self, dim: int):
        self.dim = dim
        self.W = np.zeros((dim, dim))

    def train(self, patterns: np.ndarray):
        """
        patterns: array of shape (n_patterns, dim), each entry ±1.
        Hebb rule: W = sum_p p p^T, then zero out diagonal.
        """
        self.W = patterns.T @ patterns
        np.fill_diagonal(self.W, 0)

    def recall(self, s: np.ndarray, max_iters: int = 10, synchronous=True):
        """
        s: initial state (dim,), ±1
        Returns the trajectory of states until convergence or max_iters.
        """
        traj = [s.copy()]
        for _ in range(max_iters):
            if synchronous:
                s_new = np.sign(self.W @ traj[-1])
            else:
                s_new = traj[-1].copy()
                for i in np.random.permutation(self.dim):
                    s_new[i] = np.sign(self.W[i] @ s_new)
            traj.append(s_new)
            if np.array_equal(traj[-1], traj[-2]):
                break
        return traj

    def plot_trajectory(self, traj, reshape=(5, 5), cmap="gray_r", vmin=-1, vmax=1):
        """
        Given traj (list of (dim,) arrays), produce and return a matplotlib Figure
        showing each state side by side.
        """
        import math
        T = len(traj)
        cols = min(5, T)
        rows = math.ceil(T/cols)
        fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < T:
                ax.imshow(traj[i].reshape(reshape), cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"t={i}", fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        return fig

    def animate_trajectory(self, traj, reshape=(5, 5), cmap="gray_r", vmin=-1, vmax=1,
                           figsize=(3, 3), interval=500):
        """
        Return a FuncAnimation that you can display in Jupyter (JSHTML mode).
        """
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(traj[0].reshape(reshape), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')

        def update(frame):
            im.set_data(traj[frame].reshape(reshape))
            ax.set_title(f"t={frame}", fontsize=8)
            return (im,)

        ani = animation.FuncAnimation(
            fig, update, frames=len(traj), interval=interval, blit=True, repeat=False
        )
        return ani
