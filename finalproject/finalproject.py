#Code based on example from Techvidvan
import cv2
import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf


#load in effect images
thumbs = cv.imread('thumbsup.png')
laser = cv.imread('lasershow.png')
peace = cv.imread('peacesign.png')
stop = cv.imread('stopsign.png')

#video width
vid_width = 640
vid_height = 480

#animation characterisitcs
#number of frames gesture must be held for before effect is shows
gesture_count = 10
#number of frames the effect lasts for
effect_count = 50


#import media pipe model
mp_model = mp.solutions.hands
hand_recognizer = mp_model.Hands(max_num_hands=2, min_detection_confidence=0.75)
visualizer = mp.solutions.drawing_utils

#load in gesture recognizer model
model = tf.keras.models.load_model('mp_hand_gesture')

#load in the class names
#TODO change names
f = open('gesture.names', 'r')
hand_gestures = f.read().split('\n')
f.close()
print('hand gesture options')
print(hand_gestures)

#function for altering color of image to showcase different effect moods
def tint_color(frame, red, blue, green):
    frame[:,:,0] = cv.add(frame[:,:,0], blue)
    frame[:, :, 1] = cv.add(frame[:, :, 1], green)
    frame[:, :, 2] = cv.add(frame[:, :, 2], red)
    return frame

#merges two images together with a diminishing opacity
def replace_panel(frame, image, size, center_point, count):
    weight = (count/effect_count)*0.8
    #determine if effect fits.
    width = int(size[0]/2)
    height = int(size[1]/2)
    if size[0] == 512:
        #rotate image for added fun
        rotM = cv.getRotationMatrix2D((int(512/2),int(512/2)),count,1.3)
        rot_image = cv.warpAffine(image,rotM, (512,512))
        # resize effect to full frame
        image = cv.resize(rot_image, (frame.shape[1], frame.shape[0]))
        altered_frame = cv.addWeighted(frame, 1-weight, image, weight,0)
        #return for visualization
        return altered_frame
    else:
        #if within the drawable limits
        if center_point[0] > width and center_point[1] > height:
            if center_point[0] < vid_width - width and center_point[1] < vid_height - height:
                ROI = frame[center_point[1]-width:center_point[1]+width, center_point[0]-height:center_point[0]+height,:]
                #create mask of effect
                image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                ret, mask = cv.threshold(image_gray, 10, 255, cv.THRESH_BINARY)
                mask_inv = cv.bitwise_not(mask)

                #apply mask to both images
                bg = cv.bitwise_and(ROI,ROI,mask=mask_inv)
                fg = cv.bitwise_and(image,image,mask=mask)

                #add image and effect
                #altered_ROI = cv.addWeighted(ROI, 1.0-weight, image, weight, 0.0)
                altered_ROI = cv.addWeighted(bg, 1, fg, weight, 0.0)

                frame[center_point[1]-width:center_point[1]+width, center_point[0]-height:center_point[0]+height,:] = altered_ROI

    return frame




#function for drawing the effect in the frame
def draw_effect(frame, gesture, hand_points, frame_count):
    #determine what effect to draw
    size = (100,100)
    if gesture == 'thumbs up':
        #set effect
        effect = thumbs
        #set spot effect should occur
        center = hand_points[4]

    elif gesture == 'thumbs down':
        effect = cv.flip(thumbs, 0)
        # set spot effect should occur
        center = hand_points[4]

    elif gesture == 'stop':
        # set effect
        effect = stop
        # set spot effect should occur
        center = hand_points[5]
        #pre add color tint
        frame = tint_color(frame, 40, -40, -40)

    elif gesture == 'peace':
        # set effect
        effect = peace
        # set spot effect should occur
        center = hand_points[8]
        # pre add color tint
        frame = tint_color(frame, -40, -40, 40)

    elif gesture == 'rock':
        # set effect
        effect = laser
        # set spot effect should occur
        center = (0,0)
        size = (512,512)
        #pre add color tint
        frame = tint_color(frame, -40, 40, -40)
    else:
        return frame

    #draw effect
    altered_frame = replace_panel(frame, effect, size, center, frame_count)
    #determine if size of effect will fit in location
    return altered_frame


#create camera object
vid = cv2.VideoCapture(0)
vid.set(cv.CAP_PROP_FRAME_WIDTH, 512)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 512)



#init variables for tracking current position in gesture counting and effect tracking
curr_gesture_count = 0
curr_effect_count = effect_count
last_landmarks = []
#false if not drawing effect true if
drawing_effect_flag = False
curr_gesture = 'nothing'


#camera loop
while vid.isOpened():
    # get camera frame
    ret, frame = vid.read()
    dim = frame.shape
    vid_width = dim[0]
    vid_height = dim[1]

    #convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #predict hand positions on frame
    results = hand_recognizer.process(frame_rgb)

    #initialize class name
    my_class = 'Nothing'

    #process the results
    if results.multi_hand_landmarks:
        landmarks = []
        for handslms in results.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * vid_width)
                lmy = int(lm.y * vid_height)

                landmarks.append([lmx, lmy])

            #draw landmarks on the frame
            visualizer.draw_landmarks(frame, handslms, mp_model.HAND_CONNECTIONS)

        #if not in animation phase track gesture
        if not drawing_effect_flag:
            print('detecting gesture')
            #use tf model for predictions of hand gesture
            if len(landmarks) == 21:
                predictions = model.predict([landmarks])

                classID = np.argmax(predictions)
                my_class = hand_gestures[classID]

                if my_class == curr_gesture:
                    curr_gesture_count += 1
                else:
                    curr_gesture = my_class
                    curr_gesture_count = 1

    #if in animation phase
    if drawing_effect_flag:
        #if still in animation frame of effect
        if curr_effect_count > 0:
            print('Drawing effect')
            print(curr_effect_count)
            #draw the effect and decrement animation counter
            frame = draw_effect(frame, curr_gesture, last_landmarks, curr_effect_count)
            curr_effect_count -= 1

        else:
        #reset variables for tracking gesture and animation
            drawing_effect_flag = False
            curr_effect_count = effect_count
            curr_gesture_count = 0

    #track count of continuous gesture holding
    #if true start animation phase and reset gesture count
    if curr_gesture_count >= gesture_count:
        drawing_effect_flag = True
        last_landmarks = landmarks
        curr_gesture_count = 0
        print('drawing effect')

        # show the prediction on the frame
    cv2.putText(frame, my_class, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Recognizer", frame)

    # wait for escape key to exit app
    k = cv.waitKey(10) & 0xFF
    if k == 27:
        break

# ensure video and recording are ended properly.
vid.release()
cv.destroyAllWindows()
