import cv2 as cv
import numpy as np

#load in images in list
img_names = ["IMAGE_1.JPG", "IMAGE_2.JPG", "IMAGE_3.JPG"]
img_list = [cv.imread(fn) for fn in img_names]

#use mertens to fuse images together without exposure time
merge_mertens = cv.createMergeMertens()
mertens_result = merge_mertens.process(img_list)

#convert to 8bit for showing
bin8_image = np.clip(mertens_result*255, 0 , 255).astype('uint8')

#show and save image
cv.imshow('img', bin8_image)
cv.waitKey(0)
#cv.imwrite("mertens_fuse.jpg", bin8_image)
