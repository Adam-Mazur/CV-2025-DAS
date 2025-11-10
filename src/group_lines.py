import numpy as np
import hdbscan
import math


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

    return np.linalg.norm(points1 - points2)


def custom_average(lines, width):
    points = []
    for line in lines:
        pts = to_points(line, 0, width)
        points.append(pts)

    points = np.stack(points)
    point1 = points[:, 0, :].mean(axis=0)
    point2 = points[:, 1, :].mean(axis=0)

    return from_points([point1, point2])


def group_lines(lines, width):
    n = len(lines)

    lines = np.array(lines)
    if lines.shape != (n, 2):
        raise ValueError("Input lines must be of shape (n, 2)")

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = custom_distance(lines[i], lines[j], width)
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=2)
    labels = clusterer.fit_predict(dist_matrix)

    unique_labels = np.unique(labels[labels >= 0])
    centroids = []
    for label in unique_labels:
        members = lines[labels == label]
        centroids.append(custom_average(members, width))

    return centroids
