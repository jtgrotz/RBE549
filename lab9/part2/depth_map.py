import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#load in images of left and right
imgL = cv.imread('aloeL.jpg', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('aloeR.jpg', cv.IMREAD_GRAYSCALE)

#set stereo parameters disparities was set higer which better finds the aloe plant
stereo = cv.StereoBM.create(numDisparities=64, blockSize=15)
#computes actual disparity plot
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()