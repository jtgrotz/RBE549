import cv2 as cv
import numpy as np
import time
class custom_sift:
    __intervals = 4 #intervals or number of scales
    __octaves = 5 #of octaves
    initial_sigma = np.sqrt(2)/2.0
    kernel_size = 0 #kernel for blur, set to zero to be dynamic
    r = 10 #radius for eliminating edge response
    interpolation_threshold = 0.02*256 #threshold for weak extreme
    initial_image_y = 0
    initial_image_x = 0

    #object initialization
    def __init__(self, initial_sigma):
        self.initial_sigma = initial_sigma


    #Function for setting the number of intervals
    #adjusts k value accordingly
    def set_intervals(self, intervals):
        self.__intervals = intervals

    #function for setting the number of octaves
    def set_octaves(self,octaves):
        self.__octaves = octaves

    #main SIFT function
    #input: BGR_image, threshold for feature
    #output: gray_image with features, feature locations
    def SIFT(self,image):
        #convert image to right format: GRAY_SCALE and float32 for more accurate calculations
        gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        gray_image = gray_image.astype('float32')
        shape = np.shape(gray_image)
        self.initial_image_y = shape[0]
        self.initial_image_x = shape[1]
        #cv.imshow('im',gray_image)
        #cv.waitKey(0)
        #scale_space extrema detection
        print("Creating Difference of Gaussian")
        dog = self.__create_difference_of_gaussian(gray_image)
        #accurate keypoint localization
        print("Finding Keypoints")
        keypoints = self.keypoint_localization(dog) #finds extremum within neighbors
        print("Interpolating Keypoints")
        accurate_keypoints = self.interpolate_keypoints(dog,keypoints) #uses quadratic interpolation
        print("Edge Filtering Keypoints")
        edge_filtered_keypoints = self.eliminate_edge_response(dog,accurate_keypoints) #filters edge points
        keypoint_image = self.visualize_keypoints(image,edge_filtered_keypoints)
        return keypoint_image, edge_filtered_keypoints

        #orientation assignment
        #TODO
        #keypoint_descriptor
        #TODO

        #function that creates scale space and subtracts adjacent images to form DoG
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
                #subtract neighboring images in the scale space to make the DoG must use cv.subtract
                difference = cv.subtract(scale_space[octave][s+1],scale_space[octave][s])
                #cv.imshow('im', difference)
                #cv.waitKey(0)
                DoG_octave.append(difference)
            DoG.append(DoG_octave)
        return DoG

    #function for keypoint finding and localization
    #input difference _of_gaussian
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

    #kept as a function in case there is any plan to alter keypoint before adding.
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
            # iterate until keypoint is within range or n tries have been done
            k = keypoints[k]
            oct = k[0]
            #save points a 3x1 array
            point = np.array(k[1:4], dtype=float)
            #continues to iterate if interpolation moves the pixel by a large amount for better localization
            while (value > 0.6):
                #if exceeding 5 iterations, then break.
                if iterations > 5:
                    print("reached max iterations")
                    break

                #initialize point
                #checks if point is outside of bounds, otherwise continues operation
                if self.check_outside(point,oct):
                    break
                else:
                    point = np.array(point, dtype=float)
                    #do quadratic interpolation update values = [x_prime, D_x_prime]
                    update_values = self.quadratic_interpolation(diff_of_gaussian, oct, point)
                    #determine how much interpolation moves the point
                    value = max(update_values[0])
                    if value > 0.5: #0.5 is value that would change the round operation to search at new point
                        #update (s,y,x) point with values from interpolation
                        point[0] = round(point[0]+update_values[0][0][0]) #s
                        point[1] = round(point[1]+update_values[0][1][0]) #y
                        point[2] = round(point[2]+update_values[0][2][0]) #x
                    else:
                        break
                    iterations += 1

            # if point is still within range then add to list.
            if value < 0.6:
                original_coordinate = self.convert_to_original_coordinates(point)
                #if the value of the interpolation at the given point is less than a value, discard the keypoint as unstable
                if np.absolute(update_values[1]) > self.interpolation_threshold:
                    new_keypoints.append([oct, point[0], point[1], point[2]])

        return new_keypoints



        #function form the hessian matrix
        # s is scale, y is y coordinate x is x coordinate
    def calc_hessian(self,DoG_octave,s,y,x):
        w = DoG_octave
        h11 = w[s+1][y][x]+w[s-1][y][x]-2*w[s][y][x]
        h22 = w[s][y+1][x]+w[s][y-1][x]-2*w[s][y][x]
        h33 = w[s][y][x+1]+w[s][y][x-1]-2*w[s][y][x]
        h12 = (w[s+1][y+1][x]-w[s+1][y-1][x]-w[s-1][y+1][x]+w[s-1][y-1][x])/4
        h23 = (w[s][y+1][x+1]-w[s][y+1][x-1]-w[s][y-1][x+1]+w[s][y-1][x-1])/4
        h13 = (w[s+1][y][x+1]-w[s+1][y][x-1]-w[s-1][y][x+1]+w[s-1][y][x-1])/4
        h = np.array([[h11, h12, h13], [h12, h22, h23], [h13, h23, h33]])
        return h.astype(float)

    #function approximates the 3d gradient
    # s is scale, y is y coordinate x is x coordinate
    def calc_3dgradient(self,DoG_octave,s,y,x):
        w = DoG_octave
        gs = (w[s+1][y][x]-w[s-1][y][x])/2
        gy = (w[s][y+1][x]-w[s][y-1][x])/2
        gx = (w[s][y][x+1]-w[s][y][x-1])/2
        g = np.array([[gs],[gy],[gx]])
        return g.astype(float)

    #function to compute the maximum point and extremum value using quadratic interpolation
    def quadratic_interpolation(self,DoG,oct,keypoint):
        output = []
        #convert to int for array indexing
        s = int(keypoint[0])
        y = int(keypoint[1])
        x = int(keypoint[2])
        w = np.array([[s], [y], [x]])
        gradient = self.calc_3dgradient(DoG[oct],s,y,x)
        hessian = self.calc_hessian(DoG[oct],s,y,x)
        #x_prime = H^-1 * g: distance to move point to reach true max/min
        try: #try as sometimes there is a singular matrix that cant be inverted
            x_prime = np.matmul(np.linalg.inv(hessian),gradient)
        except:
            print('Singular Matrix, skipping keypoint')
            x_prime = np.array([[0],[0],[0]])
        # D_x_prime = w + 0.5*G^T*x_prime where w is the value of the image at that point: Pixel value at new extreme
        D_x_prime = float(DoG[oct][s][y][x])-0.5*np.dot(np.matrix.transpose(gradient),x_prime)

        output.append(x_prime)
        output.append(D_x_prime)
        return output

    #function that computes eigen value ratio for edge rejection
    def eliminate_edge_response(self,DoG,keypoints):
        #intialize list
        new_keypoints = []
        #go through each keypoint and look for edge response
        for k in keypoints:
            #compute 2D hessian
            H = self.calc_2d_hessian(DoG,k)
            #compute trace(H)^2/det(H)
            ratio = (np.trace(H)**2)/np.linalg.det(H)
            #compare to value of (r+1)^2/r
            if ratio < (((self.r+1)**2)/self.r):
                new_keypoints.append(k)
        return new_keypoints

    #function for calculating 2d hessian for edge rejection
    def calc_2d_hessian(self,DoG,keypoint):
        octave = int(keypoint[0])
        s = int(keypoint[1])
        y = int(keypoint[2])
        x = int(keypoint[3])
        w = DoG[octave]
        h11 = w[s][y+1][x]+w[s][y-1][x]-2*w[s][y][x]
        h12 = (w[s][y+1][x+1]-w[s][y+1][x-1]-w[s][y-1][x+1]+w[s][y-1][x-1])/4
        h22 = w[s][y][x+1]+w[s][y][x-1]-2*w[s][y][x]
        h = np.array([[h11, h12],[h12, h22]])
        return h.astype(float)


    #funciton for drawing keypoints. Eahc keypoint is a constant size blue circle
    def visualize_keypoints(self,image,keypoints):
        for k in keypoints:
            oct = k[0]
            #adjust location of circle based on keypoints in different octaves
            x_coord = k[3] * np.power(2, oct)
            y_coord = k[2] * np.power(2, oct)
            center = (int(x_coord),int(y_coord))
            image = cv.circle(image,center,3,[255,0,0],1)

        return image

        #unsused function
    def convert_to_original_coordinates(self,keypoint):
        return None

    #function to check if interpolated point is outside of the image, so as to ignore the point.
    def check_outside(self,point, octave):
        #scales limits based on octave
        ylim = self.initial_image_y/(np.power(2,octave))
        xlim = self.initial_image_x/(np.power(2,octave))
        s_check = point[0] >= self.__intervals+1 or point[0] < 0
        y_check = point[1] < 0 or point[1] >= ylim-1
        x_check = point[2] < 0 or point[2] >= xlim-1
        return any([s_check, y_check, x_check])