import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

#load in image
bgr_image = cv.imread('UnityHall.png')
image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
shape = image.shape
#create geometric transformations
#rotation
rotation_angle = 10
center_x = (shape[1]-1)/2
center_y = (shape[0]-1)/2
rot_mat = cv.getRotationMatrix2D((center_x,center_y),rotation_angle,1)
rotated_image = cv.warpAffine(image,rot_mat, (shape[1],shape[0]))

#scale up 20%
scale_up = 1.2
scale_up_image = cv.resize(image,None,fx =scale_up, fy=scale_up, interpolation= cv.INTER_LINEAR)

#scale down 20%
scale_down = 1.2
scale_down_image = cv.resize(image,None,fx =scale_down, fy=scale_down, interpolation= cv.INTER_LINEAR)

#affine tf


#perspective tf


#plot them all together on a big subplot
plt.subplot(2,3,1), plt.imshow(image)
plt.title('Original'),plt.xticks([]),plt.yticks([])

plt.subplot(2,3,2), plt.imshow(rotated_image)
plt.title('Rotated'),plt.xticks([]),plt.yticks([])

plt.subplot(2,3,3), plt.imshow(scale_up_image)
plt.title('Scaled Up'),plt.xticks([]),plt.yticks([])

plt.subplot(2,3,4), plt.imshow(scale_down_image)
plt.title('Scaled Down'),plt.xticks([]),plt.yticks([])
#show image
plt.show()