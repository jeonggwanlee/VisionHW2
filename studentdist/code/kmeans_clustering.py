import numpy as np
from distance import pdist
import copy
from tqdm import tqdm

def kmeans_clustering(all_features, vocab_size, epsilon, max_iter):
    # Your code here. You should also change the return value.
    dim_feature = all_features.shape[1]
    num_centroids = vocab_size

    # [Desc] generate centroids using Normal distribution with (mean, 0.5*std)
    all_features_mean = np.mean(all_features, axis=0)
    all_features_std = np.std(all_features, axis=0)
    normals = []
    for mean, std in zip(all_features_mean, all_features_std):
        normal_value = np.random.normal(mean, std * 0.5, num_centroids)
        normals.append(normal_value)
    normals = np.array(normals)
    centroids = np.transpose(normals)

    # [Desc] Loop
    # 1. get centroid index per each points
    # 2. calculate each new centroid to average points which are assigned that centroid
    # 3. also collect the gap of distance between previous and present centroid
    # 4. get maximum of gap of distance to determine break the loop
    for iter in range(max_iter):

        # [Desc] ``pdist'' function calculates the distances of all features and centroids
        distances = pdist(all_features, centroids)
        # [Desc] 1. get centroid index per each points
        centroid_idxs_per_point = np.argmin(distances, axis=1)

        centroid_dist_gaps = []
        not_assigned_centroid = []
        for centroid_idx in range(num_centroids):
            points_idx = np.where(centroid_idxs_per_point == centroid_idx)
            points_idx = points_idx[0]
            if points_idx.any():
                # [Desc] 2. calcuate each enw centroid to average points which are assigned that centroid
                selected_features = all_features[points_idx]
                new_centroid = np.mean(selected_features, axis=0)
                prev_centroid = copy.deepcopy(centroids[centroid_idx])
                centroids[centroid_idx] = new_centroid
                centroid_dist_gap = np.linalg.norm(prev_centroid - new_centroid)
                # [Desc] 3. also collect the gap of distance between previous and presen centroid
                centroid_dist_gaps.append(centroid_dist_gap)
            else:
                not_assigned_centroid.append(centroid_idx)
        centroid_dist_gaps = np.array(centroid_dist_gaps)
        not_assigned_cent_str = " ".join(map(str, not_assigned_centroid))
        len_not_assigned = len(not_assigned_centroid)

        # [Desc] 4. get maximum of gap of distance to determine break the loop
        dist_max = np.max(centroid_dist_gaps)
        print('kmeans_clustering iter : {} maximum distance: {:7.3f}, not assigned centroid: {}, [{}]'\
              .format(iter, dist_max, not_assigned_cent_str, len_not_assigned))
        if dist_max < epsilon:
            break

    return centroids