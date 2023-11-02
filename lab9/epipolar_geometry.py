import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

#function that takes the two images and keypoints and finds the epipoles
#and epipolar lines
def find_epipolar(kp1, des1, img1, kp2, des2, img2, matcher):
    #find matches
    matches = matcher.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []

    #use ratio test to filter out matches
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    #find the fundamental matrix given the matches we just found
    #uses 8 point algorithm noted in lecture
    F, mask = cv.findFundamentalMat(pts1,pts2, cv.FM_RANSAC)
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    #find epilines of second image visualized on first image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    #find epilines of first image visualized on the second image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    return img3, img5



#import images

left = cv.imread('globe_left.jpg',cv.IMREAD_GRAYSCALE)
center = cv.imread('globe_center.jpg', cv.IMREAD_GRAYSCALE)
right = cv.imread('globe_right.jpg', cv.IMREAD_GRAYSCALE)


#create sift instance
sift = cv.SIFT_create()

#find sift keypoints and descriptors
kpl, desl = sift.detectAndCompute(left,None)
kpc, desc = sift.detectAndCompute(center,None)
kpr, desr = sift.detectAndCompute(right,None)

#create FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

#call function to find epilines for left and center image
lc1, lc3 = find_epipolar(kpl,desl,left,kpc,desc,center,flann)

#call function to find epilines for center and right image
cr1, cr3 = find_epipolar(kpc,desc,center,kpr,desr,right,flann)

#show images
plt.subplot(121),plt.imshow(lc1)
plt.subplot(122),plt.imshow(lc3)
plt.show()

plt.subplot(121),plt.imshow(cr1)
plt.subplot(122),plt.imshow(cr3)
plt.show()




