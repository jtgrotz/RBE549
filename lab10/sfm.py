import cv2
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
    return 100*(value[0:3].reshape(3,1))

#function for solving trangualtion using Least Squares
def LinearLSTriangulation2(u0,P0,u1,P1):
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
    B = np.array([0,0,0,1]).reshape(4,1)

    X_Prime = np.matmul(np.linalg.inv(np.matmul(A.T, A)), (np.matmul(A.T, B)))
    print(X_Prime)
    return X_Prime[0:3]

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

def reprojection_error(X,pt,P):
    new_pt = np.matmul(P, np.append(X,1).reshape(4,1))
    error = np.square(np.linalg.norm(pt[0]-new_pt[0]))+np.square(np.linalg.norm(pt[1]-new_pt[1]))
    return error

#function that triangulates each point, and counts whether it is in front of both cameras.
def check_infront(ptsl,ptsr,P0,P1):
    pts_infront = 0
    c1_error = 0
    c2_error = 0
    for i in range(len(ptsl)):
        curr_pt = LinearLSTriangulation(ptsl[i], P0, ptsr[i], P1)
        #calculate the reprojection error for each camera
        c1_error += reprojection_error(curr_pt,ptsl[i],P0)
        c2_error += reprojection_error(curr_pt, ptsr[i], P1)
        R3_C1 = P0[2, 0:3]
        T_C1 = P0[:, 3].reshape(3, 1)
        R3_C2 = P1[2,0:3]
        T_C2 = P1[:,3].reshape(3,1)
        if np.matmul(R3_C1,(curr_pt-T_C1)) > 0:
            pts_infront += 1
        if np.matmul(R3_C2,(curr_pt-T_C2)) > 0:
            pts_infront += 1
    return pts_infront, c1_error, c2_error

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


##calibrate my webcam using chessboard method.
#lab 8 camera calibration code
calibrate_camera

camera_matrix = np.load('calibration_matrix.npy')
dist_coeff = np.load('distortion_coefficients.npy')

##capture two images with webcam, create 90 degree intersection
#import images here
imgr = cv.imread('books_r.png')
imgr_gray = cv.cvtColor(imgr,cv.COLOR_BGR2GRAY)
imgl = cv.imread('books_l3.png')
imgl_gray = cv.cvtColor(imgl, cv.COLOR_BGR2GRAY)

##extract features (sift or surf) find correspondence
#create sift instance

feature_detect = cv.SIFT_create()
#feature_detect = cv.xfeatures2d.SURF_create(250)

#find sift keypoints and descriptors
kpl, desl = feature_detect.detectAndCompute(imgl_gray, None)
kpr, desr = feature_detect.detectAndCompute(imgr_gray, None)

l_features = imgl.copy()
r_features = imgr.copy()

l_features = cv.drawKeypoints(imgl,kpl,l_features,cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
r_features = cv.drawKeypoints(imgr,kpr,r_features,cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv.imwrite('imgl_feature.png',l_features)
cv.imwrite('imgr_feature.png',r_features)

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
cv.imwrite('matched_image.png',matched_image)
cv.imshow('img',matched_image)
cv.waitKey(0)

# custom ransac for finding the fundamental matrix.
iterations = 800

#sigma value for noisy epipolar constraint
sigma = 0.5
final_inlier_pts = []
my_F = []
max_inliers = 0
for i in range(iterations):
    inliers = 0
    inlier_pts = []
    rand_list = random.sample(range(len(ptsl)), 64)
    ##calculate the fundamental matrix using the 8 point algorithm
    F_temp, mask = cv.findFundamentalMat(ptsl[rand_list], ptsr[rand_list], cv.FM_8POINT)


    #verify  qr^T * F * ql = 0
    results = []
    for i in range(len(ptsl)):
        #find points
        ptl = ptsl[i]
        ptr = ptsr[i]

        #compute math
        ql = np.array([ptl[0], ptl[1], 1.0])
        qr = np.array([ptr[0], ptr[1], 1.0]).reshape(3,1)

        #epipolar constraint (ql*F*qr=0)
        x = np.matmul(ql, F_temp)
        x2 = np.matmul(x, qr)
        #check if epipolar constraint is satified to a small sigma.
        if abs(x2) <= sigma:
            inliers += 1
            inlier_pts.append(i)
            results.append(x2)
    #check to see if there are more inliers that other iterations
    if inliers > max_inliers:
        max_inliers = inliers
        final_inlier_pts = inlier_pts.copy()
        my_F = F_temp
#check to see if average and std of epipolar constraint is less than previous iteration
#    if (np.abs(np.average(results)) < min_average) and (np.std(results) < min_std):
#        min_std = np.std(results)
#        min_average = np.abs(np.average(results))
#        my_F = F_temp
print("Inliers")
print(max_inliers)
#F, mask = cv.findFundamentalMat(ptsl, ptsr, cv.FM_RANSAC,1,0.99)
F = my_F
print('Fundamental Matrix')
print(F)
print('Rank of F')
print(np.linalg.matrix_rank(F))
print('Epipolar Testing')

#filtering only good points
ptsl = ptsl[final_inlier_pts]
ptsr = ptsr[final_inlier_pts]

#visualize epilines and poles
#find epilines of second image visualized on first image
#lines1 = cv.computeCorrespondEpilines(ptsr.reshape(-1,1,2), 2,F)
#lines1 = lines1.reshape(-1,3)
#img5,img6 = drawlines(imgl_gray,imgr_gray,lines1,np.int32(ptsl),np.int32(ptsr))
#cv.imshow('right',img5)

#find epilines of first image visualized on the second image
#lines2 = cv.computeCorrespondEpilines(ptsl.reshape(-1, 1, 2), 1, F)
#lines2 = lines2.reshape(-1, 3)
#img3, img4 = drawlines(imgr_gray, imgl_gray, lines2,np.int32(ptsr), np.int32(ptsl))
#cv.imshow('left',img3)
#cv.waitKey(0)

##using the M and K of the camera, calculate the essential matrix E
#E = K' F K
KTF = np.matmul(np.transpose(camera_matrix),F)
E = np.matmul(KTF, camera_matrix)
print("Essential Matrix")
print(E)

#E2, mask = cv.findEssentialMat(ptsl,ptsr,camera_matrix,cv2.RANSAC,0.99,0.1)
#print(E2)

#verify determinant of E is 0
detE = np.linalg.det(E)
print("Determinant of E")
print(np.round(detE,3))

##extract the R and T from E using decomposition
#E = UDV^T
U,S,VT = svd(E)
print("Singular values of E")
print(S)

#refactor with ideal singular values
E_bar = np.matmul(U,np.matmul(np.array([1,0,0,0,1,0,0,0,0]).reshape(3,3),VT))
U,S,VT = svd(E_bar)
print(U)
print(S)
print(VT)

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


##triangulate the 3d Points using the linear least square triangulation technique
#test_point0 = ptsl[20]
#test_point1 = ptsr[20]

#check the cheriality constraint to see if points are in front of both cameras.
final_projection_matrix = []
max_cheriality_pts = 0
final_P = []
final_c1_error_r = 0
final_c2_error_r = 0
for p in Ps:
    curr_pts_infront, c1_reprojection_error, c2_reprojection_error = check_infront(ptsl,ptsr,P0,p)
    print(curr_pts_infront)
    ##estimate the reprojection error for both cameras
    print("Left Camera Reprojection Error")
    print(c1_reprojection_error)
    print("Right Camera Reprojection Error")
    print(c2_reprojection_error)
    if curr_pts_infront > max_cheriality_pts:
        max_cheriality_pts = curr_pts_infront
        final_projection_matrix = p
        final_c1_error_r = c1_reprojection_error
        final_c2_error_r = c2_reprojection_error

print("Final Choice Left Camera Reprojection Error")
print(c1_reprojection_error)
print("Final Choice Right Camera Reprojection Error")
print(c2_reprojection_error)

#for p in Ps:
    #my_point = LinearLSTriangulation(test_point0,P0,test_point1,p)
    #print(my_point)
    #if my_point[2] > 0:
        #final_projection_matrix = p
        #break

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

#o3d.visualization.draw_geometries([point_cloud],zoom=zoom1, front=front1, lookat=lookat1,up=up1)
o3d.visualization.draw_geometries([point_cloud])

#top view
zoom2 = 0.6999
front2 = [ 0.087476422901735171, -0.95787774863490005, -0.27352896392603882 ]
lookat2 = [ 587.08365758879688, 412.70946511723417, 15.156177125134349 ]
up2 = [ 0.012757439294041883, -0.27348181919386999, 0.96179256719579109 ]
o3d.visualization.draw_geometries([point_cloud],zoom=zoom2, front=front2, lookat=lookat2,up=up2)

