from custom_sift import custom_sift
import cv2 as cv
import numpy as np

bgr_image = cv.imread('UnityHall.png')
my_sift = custom_sift(np.sqrt(2)/2.0)

my_sift.SIFT(bgr_image)

img_cpy = bgr_image.copy()

