from custom_sift import custom_sift
import cv2 as cv
import numpy as np

bgr_image = cv.imread('lenna.png')
sigma = np.sqrt(2)/2.0;
gray_image = cv.cvtColor(bgr_image,cv.COLOR_BGR2GRAY)
#im1 = cv.GaussianBlur(gray_image,(0,0),sigma)
#im2 = cv.GaussianBlur(gray_image,(0,0),1.3*sigma)
#sub = cv.subtract(im2,im1)
#cv.imshow('img',im1)
#cv.waitKey(0)
#cv.imshow('img',im2)
#cv.waitKey(0)
#cv.imshow('img',sub)
#cv.waitKey(0)

my_sift = custom_sift(sigma)


#my_sift.set_intervals(5)
#my_sift.set_octaves(4)


im, keyp = my_sift.SIFT(bgr_image)
cv.imshow('ing',im)
cv.waitKey(0)
img_cpy = bgr_image.copy()

