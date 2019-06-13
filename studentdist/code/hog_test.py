import cv2
import ipdb
ipdb.set_trace()
hog = cv2.HOGDescriptor()
im = cv2.imread('data/train/Bedroom/image_0001.jpg')
hist1 = hog.compute(im)
hist2 = hog.compute(im, ())
ipdb.set_trace()
