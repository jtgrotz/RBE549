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
imgr = cv.imread()
imgl = cv.imread()

##extract features (sift or surf) find correspondence
#create sift instance
sift = cv.SIFT_create()

#find sift keypoints and descriptors
kpl, desl = sift.detectAndCompute(imgl,None)
kpr, desr = sift.detectAndCompute(imgr,None)

bf = cv.BFMatcher()

#match keypoints
#filter keypoints
#find homography
#then use those keypoints



##calculate the fundamental matrix using the 8 point algorithm
#verify  qr^T * F * ql = 0

##using the M and K of the camera, calculate the essential matrix E

#verify determinant of E is 0

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
