from src.preprocess import Transform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_axis(x: np.ndarray, no_labels: int = 7) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    nx = x.shape[0]
    if nx == 0:
        return np.array([], dtype=int), np.array([], dtype=object)
    npos = min(no_labels, nx)
    positions = np.linspace(0, nx - 1, npos, dtype=int)
    positions = np.unique(positions)
    labels = x[positions]
    return positions, labels


def _format_label(lbl):
    try:
        if np.issubdtype(type(lbl), np.number):
            return str(np.round(float(lbl), 3))
        if hasattr(lbl, "strftime"):
            return lbl.strftime("%H:%M:%S")
        if isinstance(lbl, (np.datetime64,)):
            return pd.to_datetime(lbl).strftime("%H:%M:%S")
    except Exception:
        pass
    return str(lbl)


def visualize_transforms(
    transforms: list[Transform], df: pd.DataFrame, save_path: str = None
):
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    frames = [df.copy()]
    names = ["Original"]
    current = df.copy()
    for t in transforms:
        current = t.apply(current)
        if isinstance(current, np.ndarray):
            current = pd.DataFrame(current)
        frames.append(current.copy())
        names.append(t.__class__.__name__)

    vmin_vmax = []
    for f in frames:
        vals = np.ravel(f.values) if f.size > 0 else np.array([])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if vmin == vmax:
                eps = abs(vmin) * 1e-6 if vmin != 0 else 1e-6
                vmin -= eps
                vmax += eps
        vmin_vmax.append((vmin, vmax))

    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, f, name, (vmin, vmax) in zip(axes, frames, names, vmin_vmax):
        im = ax.imshow(
            f.values,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(name, fontsize=10)

        ax.set_xlabel("Space (m)", fontsize=9)
        ax.set_ylabel("Time (s)", fontsize=9)

        x_arr = np.asarray(f.columns)
        y_arr = np.asarray(f.index)
        x_pos, x_lbls = set_axis(x_arr)
        y_pos, y_lbls = set_axis(y_arr)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [_format_label(lbl) for lbl in x_lbls], rotation=45, ha="right", fontsize=8
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_format_label(lbl) for lbl in y_lbls], fontsize=8)

        ax.tick_params(axis="both", which="major", labelsize=8, length=4)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.set_ylabel("Value", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
