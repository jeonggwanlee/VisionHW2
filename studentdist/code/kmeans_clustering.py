import numpy as np
from distance import pdist


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

    return np.zeros((vocab_size, all_features.shape[1]))