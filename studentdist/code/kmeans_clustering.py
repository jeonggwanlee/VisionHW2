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

    num_points = all_features.shape[0]
    dim_feature = all_features.shape[1]
    num_centroids = vocab_size

    all_features_mean = np.mean(all_features, axis=0)
    all_features_std = np.std(all_features, axis=0)
    normals = []
    for mean, std in zip(all_features_mean, all_features_std):
        normal_value = np.random.normal(mean, 0.8 * std, num_centroids)
        normals.append(normal_value)
    normals = np.array(normals)
    centroids = np.transpose(normals)

    for _ in range(max_iter):

        distances = pdist(all_features, centroids)
        centroid_idxs_per_point = np.argmin(distances, axis=1)

        centroid_dist_gaps = []
        not_assigned_centroid = []
        for centroid_idx in range(num_centroids):
            points_idx = np.where(centroid_idxs_per_point == centroid_idx)
            points_idx = points_idx[0]
            if points_idx.any():
                selected_features = all_features[points_idx]
                new_centroid = np.mean(selected_features, axis=0)
                prev_centroid = copy.deepcopy(centroids[centroid_idx])
                centroids[centroid_idx] = new_centroid
                centroid_dist_gap = np.linalg.norm(prev_centroid - new_centroid)
                centroid_dist_gaps.append(centroid_dist_gap)
            else:
                not_assigned_centroid.append(centroid_idx)
        centroid_dist_gaps = np.array(centroid_dist_gaps)
        not_assigned_cent_str = " ".join(map(str, not_assigned_centroid))

        dist_max = np.max(centroid_dist_gaps)
        print('maximum distance: {:7.3f}, not assigned centroid: {}'.format(dist_max, not_assigned_cent_str))
        if dist_max < epsilon:
            break

    return centroids