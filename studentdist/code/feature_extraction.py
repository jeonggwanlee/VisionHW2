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

        # [Desc] I guess ``win stride '' is image grid size
        winStride = (16, 16)
        padding = (0, 0)
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins,
                                deriv_aperture, win_sigma, histogram_norm_type,
                                l2_hys_threshold, gamma_correction, nlevels)

        hist = hog.compute(img, winStride, padding)

        # [Desc] reshape it as (?, 36)
        hist = np.reshape(hist, [-1, 36])
        features = hist

        """
        image_grid_size = 16
        height, width = img.shape[0:2]
        num_height = math.ceil(height / image_grid_size)
        height_range = list(range(0, num_height * image_grid_size, image_grid_size)) + [height]
        num_width = math.ceil(width / image_grid_size)
        width_range = list(range(0, num_width * image_grid_size, image_grid_size)) + [width]

        #winStride = (16, 16)
        #padding = (0, 0)

        feature_list = []
        hist2_list = []

        for h_i in range(num_height):
            for w_i in range(num_width):
                this_img = img[height_range[h_i]:height_range[h_i+1], width_range[w_i]:width_range[w_i+1]]
                #print(height_range[h_i], height_range[h_i+1], width_range[w_i], width_range[w_i+1], this_img.shape)
                hist = hog.compute(this_img)
                if type(hist) == type(np.array([1])):
                    feature_list.append(hist)
        features = np.concatenate(feature_list, 0)
        #hist2 = hog.compute(img)
        """
    elif feature == 'SIFT':
        # Your code here. You should also change the return value.

        # [Desc] To divide image into sub-images by considering ``image_grid_size'',
        # Collect each height, width indices divided by ``image_grid_size''.
        image_grid_size = 20
        height, width = img.shape[0:2]
        num_height = math.ceil(height / image_grid_size)
        height_range = list(range(0, num_height * image_grid_size, image_grid_size)) + [height]
        num_width = math.ceil(width / image_grid_size)
        width_range = list(range(0, num_width * image_grid_size, image_grid_size)) + [width]
        sift = cv2.xfeatures2d.SIFT_create()

        # [Desc] crop sub-image, apply sift extractor, collect them, and concatenate them.
        feature_list = []
        for h_i in range(num_height):
            for w_i in range(num_width):
                this_img = img[height_range[h_i]:height_range[h_i+1], width_range[w_i]:width_range[w_i+1]]
                gray = cv2.cvtColor(this_img, cv2.COLOR_BGR2GRAY)
                _, descriptors = sift.detectAndCompute(gray, None)
                if type(descriptors) == type(np.array([1])):
                    feature_list.append(descriptors)

        # [Desc] feature_list is not empty, we can concatenate, otherwise, return empty numpy array
        if len(feature_list) != 0:
            features = np.concatenate(feature_list, 0)
        else:
            features = np.array(feature_list)
        """
        #ipdb.set_trace()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        _ , descriptors = sift.detectAndCompute(gray, None)
        features = descriptors
        """
    return features




