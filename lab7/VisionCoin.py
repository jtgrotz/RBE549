import cv2 as cv
import numpy as np

#select picture here:
test_pic = cv.imread("test_coin_A.jpg")

#function to do nothing as callback for trackbar functions
def nothing(x):
    pass

def crop_image(image, ratio):
    y,x = image.shape
    center_y = int(y/2)
    center_x = int(x/2)
    delta_x = int(center_x*ratio)
    delta_y = int(center_y*ratio)

    new_image = image[(center_y-delta_y):(center_y+delta_y), (center_x-delta_x):(center_x+delta_x)]
    return new_image

#function for comparing each extracted coin to the training data and determining which coin is present
def coin_matcher(input_kp, input_des, ref_data_kp, reference_data_des, costs, matcher, threshold):
    matches = []
    cost = 0
    if len(input_kp) > 1:
        for i in range(len(reference_data_des)):
            #extract features we care about for each different coin
            ref_des = reference_data_des[i]
            ref_kp = ref_data_kp[i]
            number_of_matches = 0
            #match each trained coin to the given reference coin
            curr_match = matcher.knnMatch(ref_des, input_des, k=2)
            #ratio test for better matching
            good_matches = []
            for m,n in curr_match:
                #increment counter of good match.
                if m.distance < 0.7*n.distance:
                    good_matches.append(m)
                    number_of_matches += 1
            inliers = 0
            # if multiple matches then change into proper format for homography.
            if len(good_matches) > 8:
                src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([input_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv.findHomography(src_pts,dst_pts,cv.RANSAC, threshold)
                #inliers metric counts good matches.
                inliers = np.sum(mask)
            else:
                inliers = 0
            matches.append(inliers)
            #finds what index has the highest number of correct matches which decides what coin it is
        if np.max(matches) > 0:
            best_match = np.argmax(matches)
            cost = costs[best_match]
    return cost


#initialization
t1 = 170
t2 = 57
thresh = 5
#sift = cv.xfeatures2d.SURF_create(400)
sift = cv.SIFT_create()
vid = cv.VideoCapture(0)

#set up FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

#set up bf matcher can choose between them
bf = cv.BFMatcher()

#increase resolution of camera
vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

#create trackbars for canny threshold
cv.namedWindow('vision coin')
cv.createTrackbar('Canny Threshold 1', 'vision coin', t1, 255, nothing)
cv.createTrackbar('Accumulator Threshold', 'vision coin', t2, 200, nothing)
cv.createTrackbar('Homography Threshold', 'vision coin', thresh, 50, nothing)

#load in images of training data for coins
train_quarter = cv.imread("quarter.jpg", cv.IMREAD_GRAYSCALE)
train_penny = cv.imread("penny.jpg", cv.IMREAD_GRAYSCALE)
train_dime = cv.imread("dime.jpg", cv.IMREAD_GRAYSCALE)
train_nickel = cv.imread("nickel.jpg", cv.IMREAD_GRAYSCALE)

#train sift features on each different coin for detection
kp_quarter, des_quarter = sift.detectAndCompute(crop_image(train_quarter,0.75),None)
kp_penny, des_penny = sift.detectAndCompute(crop_image(train_penny,0.75),None)
kp_dime, des_dime = sift.detectAndCompute(crop_image(train_dime,0.75),None)
kp_nickel, des_nickel = sift.detectAndCompute(crop_image(train_nickel,0.75),None)
train_des = [des_quarter, des_penny, des_dime, des_nickel]
train_kp = [kp_quarter, kp_penny, kp_dime, kp_nickel]
costs = [0.25, 0.01, 0.1, 0.05]


#main loop for detecting coins
while vid.isOpened():
    #reads intial frame from camera and then determines zoom level and applies it
    #ret, color_frame = vid.read()
    color_frame = test_pic.copy()
    frame = cv.cvtColor(color_frame, cv.COLOR_BGR2GRAY)
    dim = frame.shape
    vid_width = dim[0]
    vid_height = dim[1]


    #Hough transform to find circles around each coin
    t1 = cv.getTrackbarPos('Canny Threshold 1', 'vision coin')
    t2 = cv.getTrackbarPos('Accumulator Threshold', 'vision coin')
    rows = frame.shape[0]
    blurred_frame = cv.GaussianBlur(frame,(7,7),1.5)
    circles_detected = cv.HoughCircles(blurred_frame, cv.HOUGH_GRADIENT, 1, rows/8, param1 = t1, param2= t2, minRadius=0, maxRadius=int(vid_width/2))
    # for each coin compare to training data and find best fit
    total_price = []
    if circles_detected is not None:
        circles = np.uint16(np.around(circles_detected))
        for c in circles[0, :]:
            # create mask around each coin for classification
            # create black base for mask
            mask = np.zeros((vid_width,vid_height), dtype="uint8")
            # draw circle with white
            cv.circle(mask, (c[0], c[1]), c[2], (255, 255, 255), -1)
            #logical and circle and image
            masked_image = cv.bitwise_and(frame, frame, mask=mask)
            #crop image of coin to exclude perimeter as it is similar on all coins
            reduced_radius = int(c[2]*0.75)
            cropped_masked_image = masked_image[c[1]-reduced_radius:c[1]+reduced_radius, c[0]-reduced_radius:c[0]+reduced_radius]
            #cv.imshow('img', cropped_masked_image)
            #find sift parameters, and compare to training data.
            kp_m, des_m = sift.detectAndCompute(cropped_masked_image, None)
            thresh = cv.getTrackbarPos('Homography Threshold', 'vision coin')
            #use coin match funciton to compare extracted features
            price = coin_matcher(kp_m, des_m, train_kp, train_des, costs, bf, float(thresh))
            total_price.append(price)


    #prints each coin, and saves total price as sum
    print(total_price)
    my_price = np.round(np.sum(total_price),2)

    #visualize circles on image
    font = cv.FONT_HERSHEY_SIMPLEX
    if circles_detected is not None:
        circles = np.uint16(np.around(circles_detected))
        circle_number = 1
        for i in circles[0,:]:
            center = (i[0], i[1])
            #draw center as number
            #cv.circle(color_frame, center, 1, (0, 100, 100), 3)
            cv.putText(color_frame,str(circle_number),center,font,3,(0,0,255),2,cv.LINE_AA)
            #draw perimeter
            radius = i[2]
            cv.circle(color_frame, center,radius,(0,0,255), 3)
            circle_number += 1
    #create box for showing price
    cv.rectangle(color_frame,(0,0),(int(vid_height/6),int(vid_width/10)),(255,0,0),-1)
    cv.putText(color_frame, "$"+str(my_price), (0,30),font,1,(255,255,255),2,cv.LINE_AA)

#show final image
    color_frame = cv.resize(color_frame, (0, 0), 0, fx=0.4, fy=0.4)
    cv.imshow('vision coin', color_frame)

    #wait for escape key to exit app
    k = cv.waitKey(1000) & 0xFF
    if k == 27:
        break

#ensure video and recording are ended properly.
vid.release()
cv.destroyAllWindows()





