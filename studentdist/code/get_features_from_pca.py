import numpy as np


def get_features_from_pca(feat_num, feature):
    """
    This function loads 'vocab_sift.npy' or 'vocab_hog.npg' file and
    returns dimension-reduced vocab into 2D or 3D.

    :param feat_num: 2 when we want 2D plot, 3 when we want 3D plot
    :param feature: 'Hog' or 'SIFT'

    :return: an N x feat_num matrix
    """

    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    # Your code here. You should also change the return value.


    return np.zeros((vocab.shape[0],2))


