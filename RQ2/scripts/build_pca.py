#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT / "cache/embeddings"
OUTPUT = ROOT / "results/figures/RQ2_representation_space.pdf"
METHODS = {
    "Random-View": EMBEDDINGS / "codebert_Random_View.npy",
    "LLM-View": EMBEDDINGS / "codebert_LLM_View.npy",
    "ContraBERT-View": EMBEDDINGS / "codebert_ContraBERT_View.npy",
    "ReCoPT-View": EMBEDDINGS / "codebert_ReCoPT_View.npy",
}
SOURCE_COLOR = "#1565C0"
VIEW_COLOR = "#D7191C"


def deterministic_indices(length: int, count: int, seed: int) -> np.ndarray:
    if count > length:
        raise ValueError(f"Cannot choose {count} from {length}")
    return np.sort(np.random.default_rng(seed).choice(length, size=count, replace=False))


def main() -> None:
    original = np.load(EMBEDDINGS / "codebert_Original.npy", allow_pickle=False)
    views = {method: np.load(path, allow_pickle=False) for method, path in METHODS.items()}
    if original.shape != (2000, 768):
        raise ValueError(f"Unexpected original embedding shape: {original.shape}")
    if any(values.shape != original.shape for values in views.values()):
        raise ValueError("View embeddings do not align with Original")

    fit_indices = deterministic_indices(len(original), 750, 456)
    fit_values = np.concatenate(
        [original[fit_indices], *[values[fit_indices] for values in views.values()]],
        axis=0,
    )
    pca = PCA(n_components=2, svd_solver="randomized", random_state=123)
    pca.fit(fit_values)

    plot_indices = deterministic_indices(len(original), 500, 987)
    original_2d = pca.transform(original[plot_indices])
    views_2d = {method: pca.transform(values[plot_indices]) for method, values in views.items()}
    combined = np.concatenate([original_2d, *views_2d.values()], axis=0)
    x_low, x_high = np.quantile(combined[:, 0], [0.005, 0.995])
    y_low, y_high = np.quantile(combined[:, 1], [0.005, 0.995])
    x_pad = (x_high - x_low) * 0.06
    y_pad = (y_high - y_low) * 0.06

    figure, axes = plt.subplots(1, 4, figsize=(12.8, 3.3))
    for axis, (method, shifted) in zip(axes, views_2d.items()):
        axis.scatter(
            original_2d[:, 0], original_2d[:, 1], s=6,
            c=SOURCE_COLOR, alpha=0.40, edgecolors="none", rasterized=False,
        )
        axis.scatter(
            shifted[:, 0], shifted[:, 1], s=8,
            c=VIEW_COLOR, alpha=0.55, edgecolors="none", rasterized=False,
        )
        axis.set_title(method, color="#333333", fontweight="bold")
        axis.set_xlim(x_low - x_pad, x_high + x_pad)
        axis.set_ylim(y_low - y_pad, y_high + y_pad)
        axis.set_xlabel("PC1")
        axis.grid(alpha=0.16, linewidth=0.5)
    axes[0].set_ylabel("PC2")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SOURCE_COLOR,
               markersize=7, label="Source program"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=VIEW_COLOR,
               markersize=7, label="Generated view"),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight")
    plt.close(figure)
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
