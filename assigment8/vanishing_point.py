import cv2 as cv
import numpy as np

#import photo
texas = cv.imread('texas.png')
cv.imshow('img',texas)
cv.waitKey(0)


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



#form least squares format
num_lines = len(lines)
#create a and b matrix shapes
a = np.zeros((num_lines,2),np.float32)
b = np.zeros((num_lines,1),np.float32)
#populate with actual values
for i in range(num_lines):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    #b = [rho_1...rho_n]'
    b[i] = rho
    #a = [cos(theta_n) sin(theta_n); ....]
    a[i][0] = np.cos(theta)
    a[i][1] = np.sin(theta)

#solve using least squares
x, r, rank, sing = np.linalg.lstsq(a,b)
print(x) #this is the solution of the least squares


#visually mark the point on the image
#convert to integer
vanish_x = int(x[0][0])
vanish_y = int(x[1][0])
#draw circle
cv.circle(texas,(vanish_x,vanish_y),12,(255,0,0),-1)
cv.imshow('img',texas)
cv.waitKey(0)