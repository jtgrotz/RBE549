from custom_sift import custom_sift
import cv2 as cv
import numpy as np

#load image
bgr_image = cv.imread('lenna.png')
#optional show original image
#cv.imshow('im',bgr_image)
#cv.waitKey(0)
#set initial sigma value for blurring
sigma = np.sqrt(2)*2

#instantiate SIFT object
my_sift = custom_sift(sigma)
#my_sift.set_octaves(4)

#call sift function, returns image and list of keypoints
im, keyp = my_sift.SIFT(bgr_image)

#show image with keypoints
cv.imshow('ing',im)
cv.waitKey(0)


