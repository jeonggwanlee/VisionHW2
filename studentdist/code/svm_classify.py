import numpy as np
from sklearn import svm
from tqdm import tqdm

def svm_classify(train_image_feats, train_labels, test_image_feats, kernel_type):
    """
    This function should train a linear SVM for every category (i.e., one vs all)
    and then use the learned linear classifiers to predict the category of every
    test image. Every test feature will be evaluated with all 15 SVMs and the
    most confident SVM will 'win'.

    :param train_image_feats: an N x d matrix, where d is the dimensionality of the feature representation.
    :param train_labels: an N array, where each entry is a string indicating the ground truth category
        for each training image.
    :param test_image_feats: an M x d matrix, where d is the dimensionality of the feature representation.
        You can assume M = N unless you've modified the starter code.
    :param kernel_type: SVM kernel type. 'linear' or 'RBF'

    :return:
        an M array, where each entry is a string indicating the predicted
        category for each test image.
    """

    categories = np.unique(train_labels)
    svc_list = []
    num_categories = len(categories)
    for cat_i in tqdm(range(num_categories)):
        category = categories[cat_i]
        svc = svm.SVC(probability=True)
        this_category_idx = np.where(train_labels == category)[0]
        new_label_for_svm = np.where(train_labels == category, 1, 0)

        svc.fit(train_image_feats, new_label_for_svm)
        predict = svc.predict(train_image_feats)
        probability = svc.predict_proba(train_image_feats)
        svc_list.append(svc)

    probability_list = []
    for cat_i in range(num_categories):
        svc = svc_list[cat_i]
        probability = svc.predict_proba(test_image_feats)[:, 1]
        probability_list.append(probability)
    probability_mat = np.array(probability_list)
    probability_mat = np.transpose(probability_mat)
    argmax_class = np.argmax(probability_mat, axis=1)

    return categories[argmax_class]