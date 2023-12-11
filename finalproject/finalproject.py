#Code based on example from Techvidvan
import cv2
import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf



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


#function for drawing the effect in the frame
def draw_effect(frame, gesture, frame_count):
    return None


#create camera object
vid = cv2.VideoCapture(0)
vid.set(cv.CAP_PROP_FRAME_WIDTH, 512)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 512)

#number of frames gesture must be held for before effect is shows
gesture_count = 10

#number of frames the effect lasts for
effect_count = 50

#init variables for tracking current position in gesture counting and effect tracking
curr_gesture_count = 0
curr_effect_count = effect_count
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

                if classID == curr_gesture:
                    curr_gesture_count += 1
                else:
                    curr_gesture = classID
                    curr_gesture_count = 1

    #if in animation phase
    if drawing_effect_flag:
        #if still in animation frame of effect
        if curr_effect_count > 0:
            print('Drawing effect')
            print(curr_effect_count)
            #draw the effect and decrement animation counter
            draw_effect(frame, curr_gesture, curr_effect_count)
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
