import cv2 as cv
import numpy as np
import calibrate_camera

##calibrate my webcam using chessboard method.
#lab 8 camera calibration code
calibrate_camera
camera_matrix = np.load('calibration_matrix.npy')
dist_coeff = np.load('distortion_coefficients.npy')

##capture two images with webcam, create 90 degree intersection
#import images here
imgr = cv.imread('books_r.png')
imgr_gray = cv.cvtColor(imgr,cv.COLOR_BGR2GRAY)
imgl = cv.imread('books_l.png')
imgl_gray = cv.cvtColor(imgl, cv.COLOR_BGR2GRAY)

##extract features (sift or surf) find correspondence
#create sift instance
#feature_detect = cv.SIFT_create()
feature_detect = cv.xfeatures2d.SURF_create(400)

#find sift keypoints and descriptors
kpl, desl = feature_detect.detectAndCompute(imgl_gray,None)
kpr, desr = feature_detect.detectAndCompute(imgr_gray,None)


#create matcher
#create FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

matcher = cv.FlannBasedMatcher(index_params, search_params)
#matcher = cv.BFMatcher()

#match keypoints
matches = matcher.knnMatch(desl,desr, k=2)
#filter keypoints
#ratio test
ptsl = []
ptsr = []
good_matches = []
for i in range(len(matches)):
    m = matches[i][0]
    n = matches[i][1]
    if m.distance <= 0.7*n.distance:
        ptsr.append(kpr[m.trainIdx].pt)
        ptsl.append(kpl[m.queryIdx].pt)
        good_matches.append([m])

ptsl = np.float32(ptsl)
ptsr = np.float32(ptsr)

matched_image = cv.drawMatchesKnn(imgl, kpl, imgr, kpr, good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow('img',matched_image)
cv.waitKey(0)
#find homography
#then use those keypoints



##calculate the fundamental matrix using the 8 point algorithm
F, mask = cv.findFundamentalMat(ptsl, ptsr, cv.FM_8POINT)
print(F)
ptsl = ptsl[mask.ravel()==1]
ptsr = ptsr[mask.ravel()==1]
#verify  qr^T * F * ql = 0
results = []
for i in range(len(ptsl)):
    #find points
    ptl = ptsl[i]
    ptr = ptsr[i]

    #compute math
    ql = np.array([ptl[0],ptl[1],1.0])
    qr = np.array([ptl[0],ptl[1],1.0])

    x = np.matmul(ql,F)
    x2 = np.matmul(x, np.transpose(qr))
    results.append(x2)

print('Min')
print(np.min(results))
print('Max')
print(np.max(results))
print('Average')
print(np.average(results))

##using the M and K of the camera, calculate the essential matrix E
#E = K' F K
KTF = np.matmul(np.transpose(camera_matrix),F)
E = np.matmul(KTF, camera_matrix)

#verify determinant of E is 0
detE = np.linalg.det(E)
print("Determinant of E")
print(np.round(detE,3))

##extract the R and T from E using decomposition

##create the projection matricies P0 and P1 for both images

##estimate the reprojection error for both cameras

##triangulate the 3d Points using the linear least square triangulation technique
# def LinearLSTriangulation()

##save to PCD file with features' dominant color
#def SavePCDToFile()
#http://www.open3d.org/docs/release/tutorial/reconstruction_system/make_fragments.html
#http://www.open3d.org/docs/release/tutorial/geometry/file_io.html
#http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html

##visualize 3d point cloud with open3D
#http://www.open3d.org/docs/release/tutorial/visualization/visualization.html
#frontal

#top

#side
