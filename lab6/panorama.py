import cv2 as cv
import numpy as np

def stitch_image(i1,i2,i3,sift):

    #convert images to gray scale
    g1 = cv.cvtColor(i1,cv.COLOR_BGR2GRAY)
    g2 = cv.cvtColor(i2, cv.COLOR_BGR2GRAY)
    g3 = cv.cvtColor(i3, cv.COLOR_BGR2GRAY)
    bf = cv.BFMatcher()

    # get final width of image
    h, w = g1.shape

    #find sift parameters for first two images, then match them
    g1_kp, g1_des = sift.detectAndCompute(g1, None)
    g2_kp, g2_des = sift.detectAndCompute(g2, None)
    match12 = bf.knnMatch(g1_des, g2_des, k=2)
    g1_pts, g2_pts = ratio_test(match12, g1_kp,g2_kp)

    #finds homography for first two images
    TF, mask = cv.findHomography(g2_pts,g1_pts,cv.RANSAC, 5.0)

    #performs warp and then adds images first two images together
    tf_i2 = cv.warpPerspective(i2,TF,(2*w, h))
    tf_i2_roi = tf_i2[0:h,0:w]
    i12 = cv.addWeighted(i1, 0.9, tf_i2_roi, 0.3, 0)
    tf_i2[0:h,0:w] = i12

    #find sift parameters for first stiched image and final image then match
    tf_g12 = cv.cvtColor(tf_i2, cv.COLOR_BGR2GRAY)
    g12_kp, g12_des = sift.detectAndCompute(tf_g12,None)
    g3_kp, g3_des = sift.detectAndCompute(g3, None)

    #matches images and performs ratio test to get rid of outliers
    match123 = bf.knnMatch(g12_des,g3_des, k=2)
    g12_pts, g3_pts = ratio_test(match123, g12_kp, g3_kp)

    #finds homography TF
    TF, mask = cv.findHomography(g3_pts, g12_pts, cv.RANSAC, 5.0)

    #warps images, and combines to final stitched images
    tf_i3 = cv.warpPerspective(i3, TF, (2 * w, h))
    tf_i3_roi = tf_i3[0:h, 0:2*w]
    i123 = cv.addWeighted(tf_i2, 0.9, tf_i3_roi, 0.3, 0)
    tf_i3[0:h, 0:2*w] = i123

    return tf_i3


def ratio_test(matches, kp1, kp2):
    #create empty list of good matches
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            #perform ratio test
            good_matches.append([m])

    #convert points to right format for finding homography transform
    pts1 = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    return pts1, pts2

#test script
if __name__ == "__main__":
    i1 = cv.imread("image_05_10_23_19_20_03.png")
    i2 = cv.imread("image_05_10_23_19_20_01.png")
    i3 = cv.imread("image_05_10_23_19_20_00.png")
    sift = cv.SIFT_create()

    ic = stitch_image(i1,i2,i3,sift)
    cv.imshow('img', ic)
    cv.waitKey(0)