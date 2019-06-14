import cv2
import numpy as np
from tqdm import tqdm
from numpy import linalg

from distance import pdist
from feature_extraction import feature_extraction


def get_bags_of_words(image_paths, feature):
    """
    This function assumes that 'vocab.mat' exists and contains an N(vocab_size) x feature vector
    length matrix 'vocab' where each row is a kmeans centroid or visual word. This
    matrix is saved to disk rather than passed in a parameter to avoid recomputing
    the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path
    :param feature: name of image feature representation.

    :return: an N(#image) x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size') below.
    """
    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    # Your code here. You should also change the return value.

    vocab_size = vocab.shape[0]
    bags_of_words = []
    for i in tqdm(range(len(image_paths))):
        path = image_paths[i]
        img = cv2.imread(path)[:, :, ::-1]

        features = feature_extraction(img, feature)
        distance_mat = pdist(features, vocab)
        centroid_pos = np.argmin(distance_mat, axis=1)

        bag_of_words = np.zeros([vocab_size])
        unique, counts = np.unique(centroid_pos, return_counts=True)
        for uniq, count in zip(unique, counts):
            bag_of_words[uniq] = count
        bag_of_words = (bag_of_words - np.mean(bag_of_words)) / np.std(bag_of_words)
        bags_of_words.append(bag_of_words)

    bags_of_words = np.array(bags_of_words)
    return bags_of_words
