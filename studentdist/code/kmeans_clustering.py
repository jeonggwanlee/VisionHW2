import numpy as np
from distance import pdist
import copy
from tqdm import tqdm

def kmeans_clustering(all_features, vocab_size, epsilon, max_iter):
    """
    The function kmeans implements a k-means algorithm that finds the centers of vocab_size clusters
    and groups the all_features around the clusters. As an output, centroids contains a
    center of the each cluster.

    :param all_features: an N x d matrix, where d is the dimensionality of the feature representation.
    :param vocab_size: number of clusters.
    :param epsilon: When the maximum distance between previous and current centroid is less than epsilon,
        stop the iteration.
    :param max_iter: maximum iteration of the k-means algorithm.

    :return: an vocab_size x d array, where each entry is a center of the cluster.
    """

    # Your code here. You should also change the return value.

    ## place k (vocab_size) centroids at random locations
    min_value = np.min(all_features)
    max_value = np.max(all_features)
    num_points = all_features.shape[0]
    dim_feature = all_features.shape[1]

    def denormalized_in_minmax(normalized, min, max):
        range = max - min
        return normalized * range + min

    centroids = []
    for _ in range(vocab_size):
        centroids.append(denormalized_in_minmax(np.random.random(dim_feature), min_value, max_value))
    centroids = np.array(centroids)

    for _ in tqdm(range(max_iter)):

        distances = pdist(all_features, centroids)
        centroid_idx_per_point = np.argmin(distances, axis=1)

        centroid_dist_gaps = []
        for centroid_idx in range(vocab_size):
            points_idx = np.where(centroid_idx_per_point == centroid_idx)
            if points_idx[0].any():
                selected_features = all_features[points_idx[0]]
                new_centroid = np.mean(selected_features, axis=0)
                prev_centroid = copy.deepcopy(centroids[centroid_idx])
                centroids[centroid_idx] = new_centroid
                centroid_dist_gap = np.linalg.norm(prev_centroid - new_centroid)
                centroid_dist_gaps.append(centroid_dist_gap)

        centroid_dist_gaps = np.array(centroid_dist_gaps)
        dist_max = np.max(centroid_dist_gaps)

        print(dist_max)
        if dist_max < epsilon:
            break

    return centroids