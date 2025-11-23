import numpy as np
from sklearn.cluster import KMeans


def cluster(stats: list[np.ndarray], n_clusters: int) -> list[int]:
    feature_vectors = []
    for segment_points in stats:
        num_points = segment_points.shape[0]
        mean_x = np.mean(segment_points[:, 0])
        mean_y = np.mean(segment_points[:, 1])
        std_x = np.std(segment_points[:, 0])
        std_y = np.std(segment_points[:, 1])
        feature_vector = [num_points, mean_x, mean_y, std_x, std_y]
        feature_vectors.append(feature_vector)

    feature_vectors = np.array(feature_vectors)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(feature_vectors)

    return labels.tolist()