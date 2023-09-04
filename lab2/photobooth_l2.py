import cv2 as cv
import numpy as np
from datetime import datetime

#Funtion definitions for main loop

#function to do nothing as callback for trackbar functions
def nothing(x):
    pass

#function for zooming in on the center of the image
def zoom_in(image, zoom_level):
    #saves original image size and zoomed image size
    s = image.shape
    h = s[0]
    w = s[1]
    zoomed_h = zoom_level * h
    zoomed_w = zoom_level * w
    center_width = int(zoomed_w/2)
    center_height = int(zoomed_h/2)

    #creates resized image of new dimensions with linear interpolation
    image = cv.resize(image,(0,0), fx=zoom_level, fy=zoom_level, interpolation=cv.INTER_LINEAR)

    #only displays the resolution of the original image with
    # a center focal point.
    image = image[int(center_height - (h/2)) : int(center_height + (h/2)),
          int(center_width - (w/2)): int(center_width + (w/2)),
          :]

    return image





#global variables
#logic flags
isrecording = False
extract_color = False
rotate_image = False
threshold_image = False
blur_image = False
flash_frame = False
pic_number = 0
vid_number = 0
rotation_angle = 10 #degrees

font = cv.FONT_HERSHEY_PLAIN
#define video capture
vid = cv.VideoCapture(0)
out = None

#define track bar for zoom and blur
cv.namedWindow('photobooth')
cv.createTrackbar('Zoom', 'photobooth',0,100, nothing)
cv.createTrackbar('BlurX','photobooth',5,30,nothing)
cv.createTrackbar('BlurY','photobooth',5,30,nothing)


#define video codec and capture
fourcc = cv.VideoWriter_fourcc(*'XVID')

#open logo image
logo = cv.imread('opencv_logo.png')
logo = cv.resize(logo,(0,0), fx = 0.14, fy = 0.14, interpolation=cv.INTER_LINEAR)
logo_shape = logo.shape

#main loop
while vid.isOpened():
    #reads intial frame from camera and then determines zoom level and applies it
    ret, frame = vid.read()
    z = (cv.getTrackbarPos('Zoom','photobooth')/10)+1
    frame = zoom_in(frame,z)
    dim = frame.shape
    vid_height = dim[0]
    vid_width = dim[1]

   #create frame copy to modify with non GUI elements.
    clean_frame = frame.copy()

    # add blur to images before adding text
    if blur_image:
        gauss_sigmax = cv.getTrackbarPos('BlurX', 'photobooth')
        gauss_sigmay = cv.getTrackbarPos('BlurY', 'photobooth')
        clean_frame = cv.GaussianBlur(clean_frame, (7, 7), sigmaX=gauss_sigmax, sigmaY=gauss_sigmay)
        frame = cv.GaussianBlur(frame, (7, 7), sigmaX=gauss_sigmax, sigmaY=gauss_sigmay)

    # add color extraction to images

    # add thresholding to images

    # add date and time to clean frame
    now = datetime.now()
    time_string = now.strftime("%d/%m/%y, %H:%M:%S")
    file_time_string = now.strftime("%d_%m_%y_%H_%M_%S")
    cv.putText(clean_frame, time_string, (int(0.7*vid_width), int(0.94*vid_height)), font, 1, (255,255,255), 1, cv.LINE_AA)

    #add ROI of date time display to upper left corner
    date_roi = clean_frame[(int(0.94*vid_height)-15):(int(0.94*vid_height)+15), (int(0.7*vid_width)-10):(int(0.7*vid_width)+170),:]
    clean_frame[0:30,vid_width-181:vid_width-1:,:] = date_roi
    #blends in opencv logo in top left
    #crop original image to size of logo
    cropped_image  = clean_frame[0:logo_shape[0], 0:logo_shape[1],:]
    blended_area = cv.addWeighted(cropped_image, 0.7, logo, 0.3,0)
    clean_frame[0:logo_shape[0], 0:logo_shape[1],:] = blended_area
    # add red constant border
    clean_frame = cv.copyMakeBorder(clean_frame, 6, 6, 6, 6, cv.BORDER_CONSTANT, value=[0, 0, 255])

    #add rotation to images


    # saves video frame here
    if isrecording:
        out.write(clean_frame)

    #flashes screen white after taking photo
    if flash_frame > 0:
        #turn image all white
        frame = np.ones(frame.shape)*255
        flash_frame -= 1


    #showing the proper frame and other GUI elements
    #basic frame at bottom
    cv.rectangle(frame,(0, int(0.8*vid_height)),(vid_width,vid_height),(90,90,90),-1)
    cv.rectangle(frame,(0, int(0.8*vid_height)),(vid_width,vid_height),(49,49,49),3)
    cv.circle(frame,(int(0.6*vid_width),int(0.9*vid_height)), int(0.05*vid_height),(194,194,194), -1)
    #recording circle changes color when recording
    if isrecording:
        record_button_color = (22,22,220)
    else:
        record_button_color = (194,194,194)
    cv.circle(frame, (int(0.4 * vid_width), int(0.9 * vid_height)), int(0.05 * vid_height), record_button_color, -1)
    #text for different buttons
    cv.putText(frame, 'Record',(int(0.35*vid_width), int(0.84*vid_height)), font, 1, (255,255,255), 1, cv.LINE_AA)
    cv.putText(frame, 'Capture',(int(0.55*vid_width), int(0.84*vid_height)), font, 1, (255,255,255), 1, cv.LINE_AA)
    cv.putText(frame, 'V',(int(0.385*vid_width),int(0.92*vid_height)),font, 2, (255,255,255), 2, cv.LINE_AA)
    cv.putText(frame, 'C',(int(0.58*vid_width),int(0.92*vid_height)),font, 2, (255,255,255), 2, cv.LINE_AA)
    cv.putText(frame, 'Press',(int(0.2*vid_width),int(0.92*vid_height)),font, 2, (255,255,255), 2, cv.LINE_AA)
    cv.imshow('photobooth', zoom_in(frame,1))


    #processing key presses for proper actions
    k = cv.waitKey(1) & 0xFF
    #check for key presses for options
    #if escape is pressed exit app
    if k == 27:
        break
    elif k == ord('c'):
        file_name = "image_" + file_time_string + ".png"
        cv.imwrite(file_name, clean_frame)
        flash_frame = 6
        pic_number += 1
    elif k == ord('v'):
        if not isrecording:
            #if not recornding then start recording
            isrecording = True
            file_name = "video_" + file_time_string + ".avi"
            out = cv.VideoWriter(file_name,fourcc, 40.0, (640,480))

        elif isrecording:
            # stop and save recording
            isrecording = False
            out.release()
    elif k == ord('e'):
        if extract_color:
            extract_color = False
        else:
            extract_color = True
    elif k == ord('r'):
        if rotate_image:
            rotate_image = False
        else:
            rotate_image = True
    elif k == ord('t'):
        if threshold_image:
            threshold_image = False
        else:
            threshold_image = True
    elif k == ord('b'):
        if blur_image:
            blur_image = False
        else:
            blur_image = True

#ensure video and recording are ended properly.
vid.release()
if out is not None:
    out.release()
cv.destroyAllWindows()


