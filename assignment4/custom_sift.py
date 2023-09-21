import cv2 as cv
import numpy as np
import time
class custom_sift:
    __intervals = 2
    __octaves = 3
    initial_sigma = np.sqrt(2)/2.0
    kernel_size = 0

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
        #cv.imshow('im',gray_image)
        #cv.waitKey(0)
        #scale_space extrema detection
        dog = self.__create_difference_of_gaussian(gray_image)
        #accurate keypoint localization
        keypoints = self.keypoint_localization(dog)
        accurate_keypoints = self.interpolate_keypoints(dog,keypoints)
        #keypoints = self.eliminate_edge_response(keypoint_image)

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
                #cv.imshow('im', new_image)
                #cv.waitKey(0)
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
                #cv.imshow('im', difference)
                #cv.waitKey(0)
                DoG_octave.append(difference)
            DoG.append(DoG_octave)
        return DoG

    def keypoint_localization(self,diff_of_gauss):
        #set storage for keypoints
        keypoints = []
        #iterate through each octave and search the 26 neighbors of each pixel and determine if max or min
        for octave in range(self.__octaves):
            for interval in range(1,self.__intervals):
                #get image size
                curr_image = diff_of_gauss[octave][interval]
                y_dim, x_dim = np.shape(curr_image)
                #search each pixel leaving space on the edge.
                for y in range(1,y_dim-1):
                    for x in range(1,x_dim-1):
                        curr_pixel = curr_image[y][x]
                        same_layer_pts = self.__getneighbors(y,x,curr_image)
                        # first check neighboring on same image
                        extremum_flag = self.__is_extremum(curr_pixel,same_layer_pts)

                        if extremum_flag:
                            # if first image dont check below, else do
                            if interval > 0:
                                below_layer_pts = self.__getneighbors(y,x,diff_of_gauss[octave][interval-1])
                                extremum_flag = self.__is_extremum(curr_pixel,below_layer_pts)

                            if extremum_flag:
                                # if last image, don't check above, else do
                                if interval < self.__intervals:
                                    above_layer_pts = self.__getneighbors(y,x,diff_of_gauss[octave][interval+1])
                                    extremum_flag = self.__is_extremum(curr_pixel, above_layer_pts)
                            #if point stayed an extremum, add to keypoints
                            if extremum_flag:
                                keypoints = self.__add_keypoint(keypoints,y,x,interval,octave)


        l = len(keypoints)
        print('Keypoints Found')
        print(l)
        return keypoints

    def eliminate_edge_response(self, keypoint_image):
        return None

    #function for comparing if point is greater than or less then all neighbors.
    def __is_extremum(self, p_initial, other_pts):
        max = all(other_pts <= p_initial)
        min = all(other_pts >= p_initial)
        return max | min

        #function for getting the nine neighbors and adding to a list
    def __getneighbors(self,y,x,image):
        points = []
        #add nine neighbor points, for the top and bottom layers.
        points.append(image[y][x])
        points.append(image[y-1][x])
        points.append(image[y+1][x])
        points.append(image[y][x-1])
        points.append(image[y][x+1])
        points.append(image[y-1][x-1])
        points.append(image[y-1][x+1])
        points.append(image[y+1][x-1])
        points.append(image[y+1][x+1])
        return points

    def __add_keypoint(self, k_list, y, x, interval, octave):
        k_list.append([octave, interval, y, x])
        return k_list


    #function to do quadratic interpolation of keypoints for subpixel accuracy
    #inputs: Difference of gaussian space, and list of keypoints
    def interpolate_keypoints(self,diff_of_gaussian,keypoints):
        new_keypoints = []
        for k in range(len(keypoints)):
            iterations = 0
            value = 1
            point = []
            # iterate until keypoint is within range or n tries have been done
            while (value > 0.6) or (iterations > 5):
                #initialize point
                point = keypoints[k]
                #do quadratic interpolation update values = [alpha, omega]
                update_values = self.quadratic_interpolation(diff_of_gaussian,point)

                #update position
                point[0] = round(point[0]+update_values[0][0][0]) #s
                point[1] = round(point[1]+update_values[0][1][0]) #y
                point[2] = round(point[2]+update_values[0][2][0]) #x
                # convert to pixel space
                value = max(update_values[0])

            if value < 0.6:
                original_coordinate = self.convert_to_original_coordinates(point)
                new_keypoints.append(point)

        #if point is still within range then add to list.

        #function form the hessian matrix
    def calc_hessian(self,DoG_octave,s,y,x):
        w = DoG_octave
        h11 = w[s+1][y][x]+w[s-1][y][x]-2*w[s][y][x]
        h22 = w[s][y+1][x]+w[s][y-1][x]-2*w[s][y][x]
        h33 = w[s][y][x+1]+w[s][y][x-1]-2*w[s][y][x]
        h12 = (w[s+1][y+1][x]-w[s+1][y-1][x]-w[s-1][y+1][x]+w[s-1][y-1][x])/4
        h23 = (w[s][y+1][x+1]-w[s][y+1][x-1]-w[s][y-1][x+1]+w[s][y-1][x-1])/4
        h13 = (w[s+1][y][x+1]-w[s+1][y][x-1]-w[s-1][y][x+1]+w[s-1][y][x-1])/4
        return np.array([[h11, h12, h13],[h12, h22, h23],[h13, h23, h33]])

    #function approximates the 3d gradient
    def calc_3dgradient(self,DoG_octave,s,y,x):
        w = DoG_octave
        gs = (w[s+1][y][x]-w[s-1][y][x])/2
        gy = (w[s][y+1][x]-w[s][y-1][x])/2
        gx = (w[s][y][x+1]-w[s][y][x-1])/2
        return np.array([[gs],[gy],[gx]])

    def quadratic_interpolation(self,DoG,keypoint):
        output = []
        oct = keypoint[0]
        s = keypoint[1]
        y = keypoint[2]
        x = keypoint[3]
        w = np.array([[s],[y],[x]])
        gradient = self.calc_3dgradient(DoG[oct],s,y,x)
        hessian = self.calc_hessian(DoG[oct],s,y,x)
        #alpha = H^-1 * g
        alpha = np.matmul(np.linalg.inv(hessian),gradient)
        # omega = w - 0.5*G^T*H^-1*g
        omega = w - 0.5*np.matmul(np.matmul(np.transpose(gradient),np.linalg.inv(hessian)),gradient)

        output.append(alpha)
        output.append(omega)
        return output

    def eliminate_edge_response(self,DoG,keypoints):
        return None

    def visualize_keypoints(self,image,keypoints):
        return None

    def convert_to_original_coordinates(self,keypoint):
        return None