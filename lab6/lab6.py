import cv2 as cv
import numpy as np

#import both pictures of boston
boston1 = cv.imread("boston1.jpeg")
boston2 = cv.imread("boston2.jpeg")
gray_b1 = cv.imread("boston1.jpeg",cv.IMREAD_GRAYSCALE)
gray_b2 = cv.imread("boston2.jpeg",cv.IMREAD_GRAYSCALE)

#get sizes of images
b1_h, b1_w = gray_b1.shape
b2_h, b2_w = gray_b2.shape

combined_w = b1_w+b2_w

#detect SIFT features
sift = cv.SIFT_create()

b1_kp, b1_des = sift.detectAndCompute(gray_b1, None)
b2_kp, b2_des = sift.detectAndCompute(gray_b2, None)

#match with Brute Force matching
bf = cv.BFMatcher()

match_boston = bf.knnMatch(b1_des,b2_des, k=2)
print(f"{len(match_boston):0f} original matches")

match_boston_good = []
#ratio test
for m,n in match_boston:
    if m.distance < 0.7*n.distance:
        match_boston_good.append([m])

print(f"{len(match_boston_good):0f} filtered matches")

im_match1 = cv.drawMatchesKnn(boston1, b1_kp, boston2, b2_kp, match_boston_good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow('img', im_match1)
cv.waitKey(0)

#convert points to the right format for homography tf
bos1_pts = np.float32([b1_kp[m[0].queryIdx].pt for m in match_boston_good]).reshape(-1,1,2)
bos2_pts = np.float32([b2_kp[m[0].trainIdx].pt for m in match_boston_good]).reshape(-1,1,2)

#use homography to find transformation on image 2 to image 1
TF, mask = cv.findHomography(bos2_pts, bos1_pts, cv.RANSAC, 5.0)


#final width should at most the combined width of the images

#apply TF to boston 2 image
tf_boston2 = cv.warpPerspective(boston2,TF,(combined_w,b1_h))


cv.imshow('img',tf_boston2)
cv.waitKey(0)
#create regions of interest for image adding.
tf_b2_roi = tf_boston2[0:b1_h, 0:b1_w]
#stitch together by weighted adding.
stiched_image_half = cv.addWeighted(boston1, 0.9, tf_b2_roi, 0.3, 0)

tf_boston2[0:b1_h, 0:b1_w] =  stiched_image_half

cv.imshow('combined', tf_boston2)
cv.waitKey(0)