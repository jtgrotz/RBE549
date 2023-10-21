import cv2 as cv
import numpy as np

#import photo
texas = cv.imread('texas.png')


#apply canny edge detection to extract edges
#initially blur image
blur_texas = cv.GaussianBlur(texas,(5,5),1)
#apply canny
canny_texas = cv.Canny(blur_texas,150,300)
cv.imshow('img',canny_texas)
cv.waitKey(0)

#run hough line transform to detect lines on edge detected image
lines = cv.HoughLines(canny_texas,1,np.pi/180,150, None, 0, 0)

#visualizes the lines for debug can comment out as needed
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(texas, pt1, pt2, (0, 0, 200), 1, cv.LINE_AA)

cv.imshow('img',texas)
cv.waitKey(0)

#filter detected lines based on angle



#form least sqaures format
np.linalg.lstsq()
#solve using SVD


#visually mark the point on the image