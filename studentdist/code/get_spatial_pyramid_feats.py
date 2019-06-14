import cv2
import numpy as np
from numpy import linalg
import math
from distance import pdist
from feature_extraction import feature_extraction
from tqdm import tqdm

def get_spatial_pyramid_feats(image_paths, max_level, feature):
    """
    This function assumes that 'vocab_hog.npy' (for HoG) or 'vocab_sift.npy' (for SIFT)
    exists and contains an N x feature vector length matrix 'vocab' where each row
    is a kmeans centroid or visual word. This matrix is saved to disk rather than passed
    in a parameter to avoid recomputing the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path,
    :param max_level: level of pyramid,
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size'), multiplies with
        (1 / 3) * (4 ^ (max_level + 1) - 1).
    """
    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    vocab_size = vocab.shape[0]
    # Your code here. You should also change the return value.
    img_sizes = []
    for i in range(len(image_paths)):
        path = image_paths[i]
        img = cv2.imread(path)[:, :, ::-1]
        img_sizes.append(list(img.shape[0:2]))
    img_sizes = np.array(img_sizes)
    max_h, max_w = np.max(img_sizes, axis=0)

    bags_of_words = []
    for i in tqdm(range(len(image_paths))):
        path = image_paths[i]
        img = cv2.imread(path)[:, :, ::-1] # rgb
        h, w = img.shape[0:2]

        imgs = []
        imgs.append(img)
        weights = [1*pow(2, max_level)]
        for level in range(1, max_level+1):
            divisor = pow(2, level)
            ratios = np.array(range(0, divisor+1)) / divisor
            h_ratios = np.ceil(h * ratios).astype(int)
            w_ratios = np.ceil(w * ratios).astype(int)
            for h_i in range(divisor):
                for w_i in range(divisor):
                    crop_img = img[h_ratios[h_i]:h_ratios[h_i+1], w_ratios[w_i]:w_ratios[w_i+1]]
                    imgs.append(crop_img)
                    weights.append(pow(2, max_level-level))

        #for each_img in imgs:
        #    print(each_img.shape)
        bag_of_words_per_image = []
        for each_idx, each_img in enumerate(imgs):
            features = feature_extraction(each_img, feature)
            bag_of_words = np.zeros([vocab_size])

            if type(features) == type(np.array([1])):
                distance_mat = pdist(features, vocab)
                features_vocab = np.argmin(distance_mat, axis=1)

                unique, counts = np.unique(features_vocab, return_counts=True)
                for uniq, count in zip(unique, counts):
                    bag_of_words[uniq] = count
                bag_of_words = bag_of_words * weights[each_idx]

            bag_of_words_per_image.append(bag_of_words)

        bag_of_words_per_image = np.array(bag_of_words_per_image).reshape([-1])
        bag_of_words_per_image = (bag_of_words_per_image - np.mean(bag_of_words_per_image)) / np.std(bag_of_words_per_image)
        bags_of_words.append(bag_of_words_per_image)

    bags_of_words = np.array(bags_of_words)
    return bags_of_words
