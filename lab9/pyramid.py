import numpy as np
import cv2 as cv

#checkerboard pattern shape
pattern_x = 7
pattern_y = 11

#define points of shape to draw
#shape_points = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
shape_points = np.float32([[0,0,0],[6,0,0],[3,6,0],[3,3,-3]])
#load in calibration matricies from previous lab
camera_matrix = np.load('calibration_matrix_original.npy')
dist_coeff = np.load('distortion_coefficients_original.npy')

#function for drawing axis a given points
def draw(img, corners, imgpts):
    corner = np.int32(tuple(corners[0].ravel()))
    img = cv.line(img, corner, np.int32(tuple(imgpts[0].ravel())), (255,0,0), 5)
    img = cv.line(img, corner, np.int32(tuple(imgpts[1].ravel())), (0,255,0), 5)
    img = cv.line(img, corner, np.int32(tuple(imgpts[2].ravel())), (0,0,255), 5)
    return img

#function for drawing triangular pyramid with green base as first three points
#and top point as the fourth point in the series.
def draw_triangular_pyramid(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    #draw green floor
    img = cv.drawContours(img, [imgpts[:3]], -1, (0,255,0), -3)

    #draw lines for sides of pyramid
    for i in range(3):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[3]),(255,0,0),3)

    return img


#set termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#prepare 10x7 grid points
objp = np.zeros((pattern_y*pattern_x,3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_x,0:pattern_y].T.reshape(-1,2)


#load image to draw on
my_img = cv.imread('IMG_6527.jpg')
gray_image = cv.cvtColor(my_img,cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray_image, (pattern_x,pattern_y), None)

if ret == True:
    corners_sub = cv.cornerSubPix(gray_image,corners,(11,11),(-1,-1),criteria)

    #find rotation and translation vectors given the previously calibrated camera parameters
    ret, rvecs, tvecs = cv.solvePnP(objp, corners_sub,camera_matrix,dist_coeff)

    #project given points to camera frame
    imgpts, jac = cv.projectPoints(shape_points, rvecs, tvecs, camera_matrix, dist_coeff)

    #run specific function for drawing pyramid
    img = draw_triangular_pyramid(my_img, corners_sub, imgpts)

    #show image a save
    cv.imshow('img', img)
    cv.imwrite('triangular.jpg',img)
    cv.waitKey(0)