import cv2 as cv
import numpy as np
class custom_filters:
    #define reccomended Scharr approximations for better
    sx_kernel = np.array([[-3,0,3],
                          [-10,0,10],
                         [-3,0,3]])
    sy_kernel = np.array([[-3,-10,-3],
                          [0,0,0],
                          [3,10,3]])
    l_kernel = np.array([[0,1,0],
                          [1,-4,1],
                          [0,1,0]])
    ld_kernel = np.array([[1,1,1],
                          [1,-8,1],
                          [1,1,1]])

    def my_sobel_filter(self, image, x_direction, y_direction, pre_blur):
        #pre blur image before operation to reduce noise
        if pre_blur:
            altered_image = cv.GaussianBlur(image,(3,3),sigmaX=1,sigmaY=1)
        else:
            altered_image = image.copy()
        if x_direction:
            k = self.sx_kernel
        elif y_direction:
            k = self.sy_kernel
        else:
            return image

        #convert image to grayscale for better sobel
        gray_image = cv.cvtColor(altered_image, cv.COLOR_BGR2GRAY)
        #flip kernel for proper convolution
        fk = cv.flip(k,-1)
        sobel_image = cv.filter2D(gray_image,cv.CV_64F,fk,borderType=cv.BORDER_CONSTANT)

        #convert to abs value and 8 bit image format
        filtered_image = cv.convertScaleAbs(sobel_image)

        return filtered_image

    def my_laplace_filter(self,image,pre_blur):
        #pre blur image before operation to reduce noise
        if pre_blur:
            altered_image = cv.GaussianBlur(image,(3,3),sigmaX=1,sigmaY=1)
        else:
            altered_image = image.copy()
        # convert image to grayscale for better sobel

        gray_image = cv.cvtColor(altered_image, cv.COLOR_BGR2GRAY)
        #kernels are x and y symmetric, so there is no need to flip
        k = self.ld_kernel
        laplace_image = cv.filter2D(gray_image,cv.CV_64F,k,borderType=cv.BORDER_CONSTANT)
        # convert to abs value and 8 bit image format
        filtered_image = cv.convertScaleAbs(laplace_image)

        return filtered_image

