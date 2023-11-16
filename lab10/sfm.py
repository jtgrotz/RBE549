import cv2 as cv
import numpy as np
import calibrate_camera
from scipy.linalg import svd
import open3d as o3d
import random

#function for saving saved points and colors as a point cloud.
#pc is a nx3 array of the x,y,z points
#color_list is a nx3 array of the same size with the rgb values
def SavePCDToFile(pc, color_list):
    #create point cloud with points
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
    #add colors to the datastruct
    p.colors = o3d.utility.Vector3dVector(color_list)
    #write to file
    o3d.io.write_point_cloud('point_cloud.pcd',p,write_ascii=True)
    #returns point cloud for visualization
    return p


#function to triangulate using the classic least squares solution.
def LinearLSTriangulation(u0,P0,u1,P1):
    #AX=B form
    #A 6x4 made from projections
    A = np.append(P0,P1)
    A = A.reshape(6,4)

    #B = [ul;ur]
    B = np.array([u0[0], u0[1], 1, u1[0], u1[1],1]).reshape(6,1)

    #least squares solution
    #(ATA)^-1 AT B
    X_Prime = np.matmul(np.linalg.inv(np.matmul(A.T, A)), (np.matmul(A.T, B)))
    #print(X_Prime)
    return X_Prime[0:3]

#function for solving trangualtion using SVD method. Not used in this implementation.
def LinearLSTriangulationSVD(u0,P0,u1,P1):
    # define each row for readability
    R1 = np.array([u0[0] * P0[2][0] - P0[0][0], u0[0] * P0[2][1] - P0[0][1], u0[0] * P0[2][2] - P0[0][2],
                   u0[0] * P0[2][3] - P0[0][3]])
    R2 = np.array([u0[1] * P0[2][0] - P0[1][0], u0[1] * P0[2][1] - P0[1][1], u0[1] * P0[2][2] - P0[1][2],
                   u0[1] * P0[2][3] - P0[1][3]])
    R3 = np.array([u1[0] * P1[2][0] - P1[0][0], u1[0] * P1[2][1] - P1[0][1], u1[0] * P1[2][2] - P1[0][2],
                   u1[0] * P1[2][3] - P1[0][3]])
    R4 = np.array([u1[1] * P1[2][0] - P1[1][0], u1[1] * P1[2][1] - P1[1][1], u1[1] * P1[2][2] - P1[1][2],
                   u1[1] * P1[2][3] - P1[1][3]])

    # create form Ax = 0
    A = np.array([R1, R2, R3, R4])

    #SVD solution
    U,S,VT = np.linalg.svd(A)

    #solution is the eigen vector corresponding to the smallest eigen value, which is the last column of Vt
    #size 4
    #value = (VT.T[:,3])
    value = (VT[:, 3])
    return value[0:3]

#function that takes a point and an image, and returns the average of the four adjacent pixel colors
def get_average_color(pt,image):
    int_pt = np.int16(pt)
    base_color = image[int_pt[1]][int_pt[0]]
    #get four neighbors
    up = np.int16(image[int_pt[1]-1][int_pt[0]])
    down = np.int16(image[int_pt[1]+1][int_pt[0]])
    left = np.int16(image[int_pt[1]][int_pt[0]-1])
    right = np.int16(image[int_pt[1]][int_pt[0]+1])

    #average each color channel
    r = np.int16((base_color[2]+up[2]+down[2]+left[2]+right[2])/5)
    g = np.int16((base_color[1]+up[1]+down[1]+left[1]+right[1])/5)
    b = np.int16((base_color[0]+up[0]+down[0]+left[0]+right[0])/5)

    return np.array([r,g,b])

#function that takes a list of corresponding points and triangulates the poisition, and finds the average color of the point in the images.
def transform_points_with_color(ptsl,imgl,ptsr,imgr,P0,P1):
    xyz_points = []
    point_colors = []
    #iterate through each correlated point and convert to xyz and rgb
    for i in range(len(ptsl)):
        curr_pt = LinearLSTriangulation(ptsl[i],P0,ptsr[i],P1)
        l_c = get_average_color(ptsl[i],imgl)
        r_c = get_average_color(ptsr[i],imgr)
        #average left and right image colors, normalize colors to be in 0-1 range
        point_colors.append(np.array([(l_c[0]+r_c[0])/2, (l_c[1]+r_c[1])/2, (l_c[2]+r_c[2])/2])/255)
        #append triangulated point
        xyz_points.append(curr_pt)

    return xyz_points, point_colors



##calibrate my webcam using chessboard method.
#lab 8 camera calibration code
calibrate_camera

camera_matrix = np.load('calibration_matrix.npy')
dist_coeff = np.load('distortion_coefficients.npy')

##capture two images with webcam, create 90 degree intersection
#import images here
imgr = cv.imread('books_r.png')
imgr_gray = cv.cvtColor(imgr,cv.COLOR_BGR2GRAY)
imgl = cv.imread('books_l.png')
imgl_gray = cv.cvtColor(imgl, cv.COLOR_BGR2GRAY)

##extract features (sift or surf) find correspondence
#create sift instance
feature_detect = cv.SIFT_create()
#feature_detect = cv.xfeatures2d.SURF_create(400)

#find sift keypoints and descriptors
kpl, desl = feature_detect.detectAndCompute(imgl_gray, None)
kpr, desr = feature_detect.detectAndCompute(imgr_gray, None)


#create matcher
#create FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

matcher = cv.FlannBasedMatcher(index_params, search_params)
#matcher = cv.BFMatcher()

#match keypoints
matches = matcher.knnMatch(desl,desr, k=2)
#filter keypoints
#ratio test
ptsl = []
ptsr = []
good_matches = []
for i in range(len(matches)):
    m = matches[i][0]
    n = matches[i][1]
    if m.distance <= 0.8*n.distance:
        ptsr.append(kpr[m.trainIdx].pt)
        ptsl.append(kpl[m.queryIdx].pt)
        good_matches.append([m])

#converts to floating point more accurate calculations.

ptsl = np.float32(ptsl)
ptsr = np.float32(ptsr)

#find and show matches between each image
matched_image = cv.drawMatchesKnn(imgl, kpl, imgr, kpr, good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow('img',matched_image)
cv.waitKey(0)

# custom ransac for finding the fundamental matrix.
iterations = 500
min_average = 1000.0
min_std = 1000.0
my_F = []
for i in range(iterations):
    rand_list = random.sample(range(len(ptsl)), 36)
    ##calculate the fundamental matrix using the 8 point algorithm
    F_temp, mask = cv.findFundamentalMat(ptsl[rand_list], ptsr[rand_list], cv.FM_8POINT)

    #ptsl = ptsl[mask.ravel()==1]
    #ptsr = ptsr[mask.ravel()==1]
    #verify  qr^T * F * ql = 0
    results = []
    for i in range(len(ptsl)):
        #find points
        ptl = ptsl[i]
        ptr = ptsr[i]

        #compute math
        ql = np.array([ptl[0], ptl[1], 1.0])
        qr = np.array([ptr[0], ptr[1], 1.0])

        #epipolar constraint (ql*F*qr=0)
        x = np.matmul(ql, F_temp)
        x2 = np.matmul(x, np.transpose(qr))
        results.append(x2)
    #check to see if average and std of epipolar constraint is less than previous iteration
    if (np.abs(np.average(results)) < min_average) and (np.std(results) < min_std):
        min_std = np.std(results)
        min_average = np.abs(np.average(results))
        my_F = F_temp

    #F, mask = cv.findFundamentalMat(ptsl, ptsr, cv.FM_RANSAC,1,0.99)
F = my_F
print('Fundamental Matrix')
print(F)
print('Rank of F')
print(np.linalg.matrix_rank(F))
print('Epipolar Testing')
print('Average')
print(np.average(results))
print('STD')
print(np.std(results))

##using the M and K of the camera, calculate the essential matrix E
#E = K' F K
KTF = np.matmul(np.transpose(camera_matrix),F)
E = np.matmul(KTF, camera_matrix)

#verify determinant of E is 0
detE = np.linalg.det(E)
print("Determinant of E")
print(np.round(detE,3))

##extract the R and T from E using decomposition
#E = UDV^T
U,S,VT = svd(E)

#Tx = [u1 x u2]x
#T is also the third column of the U matrix
#cp = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
#fm = np.matmul(U,cp)
#Tx = np.matmul(fm, np.transpose(U))
T = U[:, 2]
T = np.reshape(T,(3,1))
print('Translation Vector')
#print(Tx)
print(T)

#R = U[RM]V^T
#two different matricies to represent the two different solutions for this problem
rm = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
rm2 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
fm = np.matmul(U,rm)
fm2 = np.matmul(U,rm2)
R = np.matmul(fm,VT)
R2 = np.matmul(fm2,VT)
print("Rotation Matrix")
print(R)
print(R2)

#Test which combination of R and T put coordinates in front of camera.
#+T and R1 (1)
#+T and R2 (2)
#-T and R1 (3)
#-T and R2 (4)

##create the projection matricies P0 and P1 for both images
P0 = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]).reshape(3,4)
P1_1 = np.append(R,T,1)
P1_2 = np.append(R2,T,1)
P1_3 = np.append(R,-T,1)
P1_4 = np.append(R2,-T,1)
Ps = [P1_4,P1_2,P1_3,P1_1]

##estimate the reprojection error for both cameras

##triangulate the 3d Points using the linear least square triangulation technique
test_point0 = ptsl[20]
test_point1 = ptsr[20]
final_projection_matrix = []
for p in Ps:
    my_point = LinearLSTriangulation(test_point0,P0,test_point1,p)
    print(my_point)
    if my_point[2] > 0:
        final_projection_matrix = p
        break

#pointcloud structure is x,y,z,rgbvalue
xyz_points, color_list = transform_points_with_color(ptsl, imgl, ptsr, imgr, P0, p)

##save to PCD file with features' dominant color
point_cloud = SavePCDToFile(xyz_points, color_list)
#http://www.open3d.org/docs/release/tutorial/reconstruction_system/make_fragments.html
#http://www.open3d.org/docs/release/tutorial/geometry/file_io.html
#http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html


##visualize 3d point cloud with open3D
#http://www.open3d.org/docs/release/tutorial/visualization/visualization.html

#views obtained from ctrl c on image pane
#frontal view
zoom1 = 0.69999
front1 =  [0.019389693763850271, -0.038613667447797664, -0.9990660761240846 ]
lookat1 = [ 587.08365758879688, 412.70946511723417, 15.156177125134349 ]
up1 = [ 0.13986525728641819, -0.98932334292501301, 0.040951592739251669 ]

o3d.visualization.draw_geometries([point_cloud],zoom=zoom1, front=front1, lookat=lookat1,up=up1)

#top view
zoom2 = 0.6999
front2 = [ 0.087476422901735171, -0.95787774863490005, -0.27352896392603882 ]
lookat2 = [ 587.08365758879688, 412.70946511723417, 15.156177125134349 ]
up2 = [ 0.012757439294041883, -0.27348181919386999, 0.96179256719579109 ]
o3d.visualization.draw_geometries([point_cloud],zoom=zoom2, front=front2, lookat=lookat2,up=up2)

