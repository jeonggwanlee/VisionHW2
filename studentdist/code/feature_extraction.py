import cv2
import numpy as np
import math

def feature_extraction(img, feature):
    """
    This function computes defined feature (HoG, SIFT) descriptors of the target image.

    :param img: a height x width x channels matrix,
    :param feature: name of image feature representation.

    :return: a N x feature_size matrix.
    """

    if feature == 'HoG':
        # HoG parameters
        win_size = (32, 32)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        deriv_aperture = 1
        win_sigma = 4
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 64

        # Your code here. You should also change the return value.

        ## and use an image grid size 16 for HOG.
        ## and if we run HOG then it becomes 36.

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins,
                                deriv_aperture, win_sigma, histogram_norm_type,
                                l2_hys_threshold, gamma_correction, nlevels)
        winStride = (16, 16)
        padding = (0, 0)
        #locations = ((10, 20),)
        hist = hog.compute(img, winStride, padding)
        a1 = math.floor((img.shape[0]-win_size[0]) / winStride[0]) + 1
        b1 = math.floor((img.shape[1]-win_size[1]) / winStride[1]) + 1
        size = a1 * b1 * 36
        hist = np.reshape(hist, [-1, 36])
        return hist

    elif feature == 'SIFT':

        # Your code here. You should also change the return value.

        ## When you implement SIFT, use an image grid size 20.
        ## we run SIFT feature extractor with parameters given in the starter code
        ## then the feature dimension becomes 128, and

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        _, descriptors = sift.detectAndCompute(gray, None)
        # try:
        #     pad = np.zeros([2000 - descriptors.shape[0], 128])
        # except:
        #     print(descriptors.shape[0])
        #     import ipdb; ipdb.set_trace()
        # descriptors = np.concatenate([descriptors, pad], axis=0)
        return descriptors




