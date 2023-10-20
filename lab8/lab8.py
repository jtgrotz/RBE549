import numpy as np
import cv2 as cv
import glob

#set termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#prepare 10x7 grid points
objp = np.zeros((11*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:11].T.reshape(-1,2)

#arrays to store points from all images
objpoints = []
imgpoints = []

#find image names
images = glob.glob('calibration_data/*.jpg')

for iname in images:
    img = cv.imread(iname)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #find corners
    ret, corners = cv.findChessboardCorners(gray_img,(7,11), None)

    #add points to object points
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray_img, corners,(11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        #draw and display corners
        cv.drawChessboardCorners(img,(7,11), corners2,ret)
        cv.imshow('img',img)
        cv.waitKey(200)

#determine calibration values
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints,gray_img.shape[::-1], None, None)

print("camera matrix")
print(mtx)
print("distortion coefficients")
print(dist)
#print("rotation vectors")
#print(rvecs)
#print("translation vectors")
#print(tvecs)
np.save("calibration_matrix", mtx)
np.save("distortion_coefficients", dist)
#undistortion

#reference image
image = cv.imread('calibration_data/IMG_6528.jpg')
h,w = image.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

#undistort
un_dist = cv.undistort(image,mtx,dist,None,newcameramtx)

#crop the image based on distorition
x, y, w, h, = roi
un_dist = un_dist[y:y+h, x:x+w]
#cv.imwrite('calibration_result.jpg',un_dist)
cv.imshow('undist',un_dist)
cv.waitKey(0)


