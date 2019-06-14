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
    vocab_size = vocab.shape[0]
    #cov1 = np.cov(vocab)
    #w1, v1 = np.linalg.eig(cov1)

    vocab_mean = np.mean(vocab, axis=0)
    vocab = vocab - np.tile(vocab_mean, (vocab_size, 1))
    #cov2 = np.cov(vocab)
    #w2, v2 = np.linalg.eig(cov2)

    vocab = vocab / np.std(vocab, axis=0)
    cov = np.cov(vocab)
    eig_value, eig_vector = np.linalg.eig(cov)

    eig_vectors = []
    for feat_idx in range(feat_num):
        eig_vectors.append(np.real(eig_vector[feat_idx]))
    eig_vectors = np.transpose(eig_vectors)

    return eig_vectors


