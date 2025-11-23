from src.transforms import Transform
from matplotlib.colors import Normalize
import matplotlib as mpl
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


def visualize_dataframe(ax, df: pd.DataFrame, name: str = "DAS data"):
    low, high = np.percentile(df, [1, 99])
    norm = Normalize(vmin=low, vmax=high, clip=True)
    im = ax.imshow(df, interpolation="none", aspect="auto", norm=norm)
    plt.colorbar(im, ax=ax)
    ax.set_title(name, fontsize=10)
    ax.set_ylabel("Time")
    ax.set_xlabel("Space [m]")
    x_positions, x_labels = set_axis(df.columns)
    x_labels = x_labels.astype(float)
    ax.set_xticks(x_positions, np.round(x_labels))
    y_positions, y_labels = set_axis(df.index.time)
    ax.set_yticks(y_positions, y_labels)


def visualize_transforms(
    transforms: list[Transform], df: pd.DataFrame, save_path: str = None
):
    dataframes = [df.copy()]
    names = ["Original"]
    current = df.copy()
    for t in transforms:
        current = t.apply(current)
        dataframes.append(current.copy())
        names.append(t.__class__.__name__)

    n = len(dataframes)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), constrained_layout=True)

    for ax, df, name in zip(axes, dataframes, names):
        visualize_dataframe(ax, df, name)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def visualize_lines(image: pd.DataFrame, lines: list[tuple], save_path: str = None):
    _ = plt.figure(figsize=(6, 6))
    ax = plt.axes()

    visualize_dataframe(ax, image, name="Detected Lines")

    for line in lines:
        if len(line) == 2:
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x1 = 1
            x2 = image.shape[1] - 1
            y1 = (rho - a * x1) / b
            y2 = (rho - a * x2) / b
            y1 = np.clip(y1, 0, image.shape[0] - 1)
            y2 = np.clip(y2, 0, image.shape[0] - 1)
            ax.plot([x1, x2], [y1, y2], color="red", linewidth=2)
        elif len(line) == 4:
            x1, y1, x2, y2 = line
            ax.plot([x1, x2], [y1, y2], color="red", linewidth=2)

        velocity = (x2 - x1) / (y2 - y1) if y2 != y1 else float("inf")
        dx = image.columns[1] - image.columns[0]
        dt = (image.index[1] - image.index[0]).total_seconds()
        velocity = velocity * (dx / dt) * 3.6  # Convert to km/h
        velocity = abs(velocity)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(
            mid_x,
            mid_y,
            f"{velocity:.2f} km/h",
            color="yellow",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def visualize_clusters(
    image: pd.DataFrame, lines: list[tuple], labels: list[int], save_path: str = None
):
    _ = plt.figure(figsize=(6, 6))
    ax = plt.axes()

    unique_labels = set(labels)
    cmap = mpl.cm.get_cmap("vanimo", len(unique_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}

    visualize_dataframe(ax, image, name="Detected Lines")

    for line, label in zip(lines, labels):
        if len(line) == 2:
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x1 = 1
            x2 = image.shape[1] - 1
            y1 = (rho - a * x1) / b
            y2 = (rho - a * x2) / b
            y1 = np.clip(y1, 0, image.shape[0] - 1)
            y2 = np.clip(y2, 0, image.shape[0] - 1)
            ax.plot([x1, x2], [y1, y2], color=label_to_color[label], linewidth=2)
        elif len(line) == 4:
            x1, y1, x2, y2 = line
            ax.plot([x1, x2], [y1, y2], color=label_to_color[label], linewidth=2)

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(
            mid_x,
            mid_y,
            f"{label}",
            color="yellow",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def visualize_polynomial(
    image: pd.DataFrame, lines: list[tuple], labels: list[int], save_path: str = None
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    unique_labels = set(labels)
    cmap = mpl.cm.get_cmap("tab20", len(unique_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}

    visualize_dataframe(axes[0], image, name="Detected Lines")

    dx = image.columns[1] - image.columns[0]
    dt = (image.index[1] - image.index[0]).total_seconds()

    axes[1].set_title("Velocity over time")
    axes[1].set_ylabel("Velocity [km/h]")
    axes[1].set_xlabel("Time [s]")

    for line, label in zip(lines, labels):
        a, b, c, d, y1, y2 = line

        y = np.linspace(y1, y2 + 1, 1000)
        x = a * y**3 + b * y**2 + c * y + d
        der = 3 * a * y**2 + 2 * b * y + c
        w = image.shape[1]
        mask = (x >= 0) & (x <= w - 1)
        if not np.any(mask):
            continue

        x_valid = x[mask]
        y_valid = y[mask]
        der_valid = der[mask] * (dx / dt) * 3.6  # convert to km/h

        axes[0].plot(x_valid, y_valid, color=label_to_color[label], linewidth=2)
        axes[1].plot(y_valid * dt, der_valid, color=label_to_color[label], linewidth=2)

        y1 = y_valid[0]
        y2 = y_valid[-1]
        x1 = np.clip(x_valid[0], 0, w - 1)
        x2 = np.clip(x_valid[-1], 0, w - 1)
        der1 = der_valid[0]
        der2 = der_valid[-1]

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        axes[0].text(
            mid_x,
            mid_y,
            f"{label}",
            color="yellow",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

        mid_der = (der1 + der2) / 2
        axes[1].text(
            mid_y * dt,
            mid_der,
            f"{label}",
            color="yellow",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
