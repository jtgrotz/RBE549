import cv2 as cv
import numpy as np
import time
class custom_sift:
    __intervals = 2
    __octaves = 3
    initial_sigma = np.sqrt(2)/2.0
    kernel_size = 5

    def __init__(self, initial_sigma):
        self.initial_sigma = initial_sigma


    #Function for setting the number of intervals
    #adjusts k value accordingly
    def set_intervals(self, intervals):
        return None

    #function for setting the number of octaves
    def set_octaves(self,octaves):
        return None

    #main SIFT function
    #input: BGR_image, threshold for feature
    #output: gray_image with features, feature locations
    def SIFT(self,image):
        #convert image to right format: GRAY_SCALE
        gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        cv.imshow('im',gray_image)
        cv.waitKey(0)
        #scale_space extrema detection
        #scale_space = self.create_scale_space(gray_image)
        #dog = self.difference_of_gaussian(scale_space)
        dog = self.__create_difference_of_gaussian(gray_image)
        #accurate keypoint localization
        keypoint_image = self.keypoint_localization(dog)
        keypoints = self.eliminate_edge_response(keypoint_image)

        #orientation assignment
        #TODO
        #keypoint_descriptor
        #TODO

    def __create_difference_of_gaussian(self,image):
        #define k as 2^(1/s)
        k = np.power(2,(1/self.__intervals))
        sampled_image = image.copy()
        #set up storage for s+3 images per octave
        scale_space = []
        for octave in range(self.__octaves):
            scale_space_octave = []
            base_sigma = np.power(2,octave)*self.initial_sigma
            #downsample image if not the first octave
            if octave > 0:
                sampled_image = cv.pyrDown(sampled_image)

            #s+3 images in the stack to make the DoG which has s+2 difference functions
            for interval in range (self.__intervals+3):
                sigma_prime = np.power(k,interval)*base_sigma
                #blur image based on calculated sigma
                new_image = cv.GaussianBlur(sampled_image,(self.kernel_size,self.kernel_size),sigma_prime)
                cv.imshow('im', new_image)
                cv.waitKey(0)
                scale_space_octave.append(new_image)
            scale_space.append(scale_space_octave)

        #create dog space
        #initial storage
        DoG = []
        for octave in range(self.__octaves):
            DoG_octave = []
            for s in range(self.__intervals+2):
                #subtract neighboring images in the scale space to make the DoG
                difference = cv.subtract(scale_space[octave][s+1],scale_space[octave][s])
                cv.imshow('im', difference)
                cv.waitKey(0)
                DoG_octave.append(difference)
            DoG.append(DoG_octave)
        return DoG

    def keypoint_localization(self,diff_of_gauss):
        return None

    def eliminate_edge_response(self, keypoint_image):
        return None