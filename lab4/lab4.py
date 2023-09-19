import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

#load in image
bgr_image = cv.imread('UnityHall.png')
image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
shape = image.shape
#create geometric transformations
#rotation
rotation_angle = 10
center_x = (shape[1]-1)/2
center_y = (shape[0]-1)/2
rot_mat = cv.getRotationMatrix2D((center_x,center_y),rotation_angle,1)
rotated_image = cv.warpAffine(image,rot_mat, (int(shape[1]),int(shape[0])))

#scale up 20%
scale_up = 1.2
scale_up_image = cv.resize(image,(int(shape[1]*scale_up),int((shape[0]*scale_up))), interpolation= cv.INTER_LINEAR)

#scale down 20%
scale_down = 0.8
scale_down_image = cv.resize(image,(0,0),fx =scale_down, fy=scale_down, interpolation= cv.INTER_LINEAR)

#affine tf
w = shape[1]
h = shape[0]
initial_pts = np.float32([[0.2*w,0.2*h],[0.2*w, 0.8*h],[0.8*w, 0.8*h]])
final_pts = np.float32([[0.2*w,0.3*h],[0.2*w, 0.9*h],[0.8*w, 0.8*h]])
tf = cv.getAffineTransform(initial_pts,final_pts)
affine_image = cv.warpAffine(image, tf, (int(w*1.2),int(h*1.2)))

#perspective tf
initial_p_pts = np.float32([[0.2*w, 0.2*h], [0.8*w, 0.2*h], [0.2*w, 0.8*h], [0.8*w, 0.7*h]])
final_p_pts = np.float32([[0, 0], [0.9*w, 0.1*h], [0.1*w, 0.9*h], [0.9*w, 0.9*h]])
tf_p = cv.getPerspectiveTransform(initial_p_pts, final_p_pts)
perspective_image = cv.warpPerspective(image, tf_p, (int(w*1.3), int(h*1.3)))

#plot them all together on a big subplot
fig, ax = plt.subplots(2,3, figsize=(15,7), sharex=True, sharey=True)
fig.set_facecolor("black")
fig.suptitle("Geometric Transforms", color='white')

ax[0][0].imshow(image)
ax[0][0].set_facecolor("black")
ax[0][0].set_title('Original', color='white'),plt.xticks([]),plt.yticks([])

ax[0][1].imshow(rotated_image)
ax[0][1].set_facecolor("black")
ax[0][1].set_title('Rotated', color='white'),plt.xticks([]),plt.yticks([])

ax[0][2].imshow(scale_up_image)
ax[0][2].set_facecolor("black")
ax[0][2].set_title('Scaled Up', color='white'),plt.xticks([]),plt.yticks([])

ax[1][0].imshow(scale_down_image)
ax[1][0].set_facecolor("black")
ax[1][0].set_title('Scaled Down', color='white'),plt.xticks([]),plt.yticks([])

ax[1][1].imshow(affine_image)
ax[1][1].set_facecolor("black")
ax[1][1].set_title('Affine', color='white'),plt.xticks([]),plt.yticks([])

ax[1][2].imshow(perspective_image)
ax[1][2].set_facecolor("black")
ax[1][2].set_title('Perspective', color='white'),plt.xticks([]),plt.yticks([])

plt.show()


##part 2
#harris corner detection
#images need to be in gray scale and float 32 format
#parameters for corner detection
block_size = 5
kernel_size = 5
k = 0.1
c = 0.01

#original
gray_org = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
gray_orgf32 = np.float32(gray_org)
corner_org = cv.cornerHarris(gray_orgf32,block_size,kernel_size, k)
img_cpy = image.copy()
img_cpy[corner_org>c*corner_org.max()] = [255,0,0]

#rotated
gray_rot = cv.cvtColor(rotated_image,cv.COLOR_BGR2GRAY)
gray_rotf32 = np.float32(gray_rot)
corner_rot = cv.cornerHarris(gray_rotf32,block_size,kernel_size, k)
rot_cpy = rotated_image.copy()
rot_cpy[corner_rot>c*corner_rot.max()] = [255,0,0]

#scaled up
gray_sup = cv.cvtColor(scale_up_image,cv.COLOR_BGR2GRAY)
gray_supf32 = np.float32(gray_sup)
corner_sup = cv.cornerHarris(gray_supf32,block_size,kernel_size, k)
sup_cpy = scale_up_image.copy()
sup_cpy[corner_sup>c*corner_sup.max()] = [255,0,0]

#scaled down
gray_sdwn = cv.cvtColor(scale_down_image,cv.COLOR_BGR2GRAY)
gray_sdwnf32 = np.float32(gray_sdwn)
corner_sdwn = cv.cornerHarris(gray_sdwnf32,block_size,kernel_size, k)
sdwn_cpy = scale_down_image.copy()
sdwn_cpy[corner_sdwn>c*corner_sdwn.max()] = [255,0,0]

#affine
gray_aff = cv.cvtColor(affine_image,cv.COLOR_BGR2GRAY)
gray_afff32 = np.float32(gray_aff)
corner_aff = cv.cornerHarris(gray_afff32,block_size,kernel_size, k)
aff_cpy = affine_image.copy()
aff_cpy[corner_aff>c*corner_aff.max()] = [255,0,0]


#perspective
gray_per = cv.cvtColor(perspective_image,cv.COLOR_BGR2GRAY)
gray_perf32 = np.float32(gray_per)
corner_per = cv.cornerHarris(gray_perf32,block_size,kernel_size, k)
corner_per = cv.dilate(corner_per, None)
p_cpy = perspective_image.copy()
p_cpy[corner_per>c*corner_per.max()] = [255,0,0]



fig, ax = plt.subplots(2,3, figsize=(15,9), sharex=True, sharey=True)
fig.set_facecolor("black")
fig.suptitle("Harris Corners", color='white')

ax[0][0].imshow(img_cpy)
ax[0][0].set_facecolor("black")
ax[0][0].set_title('Original', color='white'),plt.xticks([]),plt.yticks([])

ax[0][1].imshow(rot_cpy)
ax[0][1].set_facecolor("black")
ax[0][1].set_title('Rotated', color='white'),plt.xticks([]),plt.yticks([])

ax[0][2].imshow(sup_cpy)
ax[0][2].set_facecolor("black")
ax[0][2].set_title('Scaled Up', color='white'),plt.xticks([]),plt.yticks([])

ax[1][0].imshow(sdwn_cpy)
ax[1][0].set_facecolor("black")
ax[1][0].set_title('Scaled Down', color='white'),plt.xticks([]),plt.yticks([])

ax[1][1].imshow(aff_cpy)
ax[1][1].set_facecolor("black")
ax[1][1].set_title('Affine', color='white'),plt.xticks([]),plt.yticks([])

ax[1][2].imshow(p_cpy)
ax[1][2].set_facecolor("black")
ax[1][2].set_title('Perspective', color='white'),plt.xticks([]),plt.yticks([])

#show image
plt.show()

##SIFT detection
#create sift object
sift = cv.SIFT_create()

#original
k_orig = sift.detect(gray_org,None)
s_orig = image.copy()
s_orig = cv.drawKeypoints(image, k_orig, s_orig, color=[0,255,0])

#rotated
k_rot = sift.detect(gray_rot,None)
s_rot = rotated_image.copy()
s_rot = cv.drawKeypoints(rotated_image,k_rot,s_rot, color=[0,255,0])

#scaled up
k_sup = sift.detect(gray_sup,None)
s_sup = scale_up_image.copy()
s_sup = cv.drawKeypoints(scale_up_image,k_sup,s_sup,color=[0,255,0])

#scaled down
k_dwn = sift.detect(gray_sdwn,None)
s_dwn = scale_down_image.copy()
s_dwn = cv.drawKeypoints(scale_down_image,k_dwn,s_dwn,color=[0,255,0])

#affine
k_aff = sift.detect(gray_aff,None)
s_aff = affine_image.copy()
s_aff = cv.drawKeypoints(affine_image,k_aff,s_aff,color=[0,255,0])

#perspective
k_per = sift.detect(gray_per,None)
s_per = perspective_image.copy()
s_per = cv.drawKeypoints(perspective_image,k_per,s_per, color=[0,255,0])

#Subplot plotting
fig, ax = plt.subplots(2,3, figsize=(15,9), sharex=True, sharey=True)
fig.set_facecolor("black")
fig.suptitle("SIFT Features", color='white')

ax[0][0].imshow(s_orig)
ax[0][0].set_facecolor("black")
ax[0][0].set_title('Original', color='white'),plt.xticks([]),plt.yticks([])

ax[0][1].imshow(s_rot)
ax[0][1].set_facecolor("black")
ax[0][1].set_title('Rotated', color='white'),plt.xticks([]),plt.yticks([])

ax[0][2].imshow(s_sup)
ax[0][2].set_facecolor("black")
ax[0][2].set_title('Scaled Up', color='white'),plt.xticks([]),plt.yticks([])

ax[1][0].imshow(s_dwn)
ax[1][0].set_facecolor("black")
ax[1][0].set_title('Scaled Down', color='white'),plt.xticks([]),plt.yticks([])

ax[1][1].imshow(s_aff)
ax[1][1].set_facecolor("black")
ax[1][1].set_title('Affine', color='white'),plt.xticks([]),plt.yticks([])

ax[1][2].imshow(s_per)
ax[1][2].set_facecolor("black")
ax[1][2].set_title('Perspective', color='white'),plt.xticks([]),plt.yticks([])

plt.show()


