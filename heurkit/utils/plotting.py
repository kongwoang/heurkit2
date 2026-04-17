"""
Plotting helpers (optional — requires matplotlib).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heurkit.core.result import SearchResult


def plot_convergence(
    results: list[SearchResult],
    title: str = "Convergence",
    save_path: str | None = None,
) -> None:
    """Plot best-so-far curves for one or more search results.

    Parameters
    ----------
    results : list[SearchResult]
        Results to plot.
    title : str
        Plot title.
    save_path : str or None
        If given, save the figure to this path instead of showing it.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        if r.history:
            ax.plot(r.history, label=r.algorithm_name)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Objective")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
