from sklearn.linear_model import LinearRegression
import warnings
import numpy as np
import hdbscan
import math

warnings.filterwarnings("ignore", message=".*force_all_finite.*")


def to_points(line, x_start, x_end):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)

    if b == 0:
        b = 1e-6

    y_0 = (rho - a * x_start) / b
    y_1 = (rho - a * x_end) / b

    points = np.array([[x_start, y_0], [x_end, y_1]])

    return points


def from_points(points):
    x1, y1 = points[0]
    x2, y2 = points[1]

    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy)
    theta = math.atan2(-dx, dy)
    rho = (x1 * dy - y1 * dx) / L

    if rho < 0:
        rho = -rho
        theta += math.pi

    return np.array([rho, theta])


def custom_distance(line1, line2, width):
    points1 = to_points(line1, 0, width)
    points2 = to_points(line2, 0, width)

    return np.linalg.norm(points1 - points2, axis=1).max()


def custom_average(lines, width):
    points = []
    for line in lines:
        pts = to_points(line, 0, width)
        points.append(pts)

    points = np.stack(points)
    point1 = points[:, 0, :].mean(axis=0)
    point2 = points[:, 1, :].mean(axis=0)

    return from_points([point1, point2])


def group_lines(lines, width, min_cluster_size=2):
    n = len(lines)

    lines = np.array(lines)
    if lines.shape != (n, 2):
        raise ValueError("Input lines must be of shape (n, 2)")

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = custom_distance(lines[i], lines[j], width)
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        allow_single_cluster=True,
    )
    labels = clusterer.fit_predict(dist_matrix)

    unique_labels = np.unique(labels[labels >= 0])
    centroids = []
    for label in unique_labels:
        members = lines[labels == label]
        centroids.append(custom_average(members, width))

    return centroids


def group_segments(segments, min_cluster_size=2, min_samples=1, points_per_segment=10):
    n = len(segments)

    segments = np.array(segments)
    if segments.shape != (n, 4):
        raise ValueError("Input segments must be of shape (n, 4)")

    points = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        t = np.random.rand(points_per_segment)
        x_random = x1 + t * (x2 - x1)
        y_random = y1 + t * (y2 - y1)
        pts = np.stack([x_random, y_random], axis=1)
        points.append(pts)

    points = np.concatenate(points)

    clusterer = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        allow_single_cluster=True,
    )
    labels = clusterer.fit_predict(points)

    unique_labels = np.unique(labels[labels >= 0])
    centroids = []
    for label in unique_labels:
        members = points[labels == label]
        X = members[:, 0].reshape(-1, 1)
        y = members[:, 1]

        model = LinearRegression()
        model.fit(X, y)

        x1 = X.min()
        x2 = X.max()
        y1 = model.predict([[x1]])[0]
        y2 = model.predict([[x2]])[0]
        centroids.append(np.array([x1, y1, x2, y2]))

    return centroids
