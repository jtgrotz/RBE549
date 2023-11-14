import cv2 as cv
import numpy as np
import calibrate_camera
from scipy.linalg import svd
import open3d as o3d
import open3d.core as o3c

def SavePCDToFile(pc):
    xyzi = np.random.rand(100, 4)

    xyz = xyzi[:, 0:3]
    i = [[i] for i in xyzi[:, 3]]
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    p.colors = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud('test.pcd',p,write_ascii=True)
    return 0

def test_points_in_front(pts1,pts2,R,T):
    return 0

def LinearLSTriangulation(u0,P0,u1,P1):
    #define each row for readability
    R1 = np.array([ u0[0]*P0[2][0]-P0[1][0], u0[0]*P0[2][1]-P0[1][1], u0[0]*P0[2][2]-P0[1][2], u0[0]*P0[2][3]-P0[1][3]])
    R2 = np.array([ u0[1]*P0[2][0]-P0[1][0], u0[1]*P0[2][1]-P0[1][1], u0[1]*P0[2][2]-P0[1][2], u0[1]*P0[2][3]-P0[1][3]])
    R3 = np.array([ u1[0]*P1[2][0]-P1[1][0], u1[0]*P1[2][1]-P1[1][1], u1[0]*P1[2][2]-P1[1][2], u1[0]*P1[2][3]-P1[1][3]])
    R4 = np.array([ u1[1]*P1[2][0]-P1[1][0], u1[1]*P1[2][1]-P1[1][1], u1[1]*P1[2][2]-P1[1][2], u1[1]*P1[2][3]-P1[1][3]])

    #create form Ax = 0
    A = np.array([R1,R2,R3,R4])

    #least squares solution
    X_Prime = np.matmul(np.linalg.inv(np.matmul(A.T, A)), (np.matmul(A.T, np.array([[0], [0], [0], [0]]))))
    print(X_Prime)
    return X_Prime

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
    #print(U)
    #print(S)
    print('VT')
    print(VT)

    #solution is the eigen vector corresponding to the smallest eigen value, which is the last column of Vt
    return (VT.T[:,3])

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
#feature_detect = cv.SIFT_create()
feature_detect = cv.xfeatures2d.SURF_create(400)

#find sift keypoints and descriptors
kpl, desl = feature_detect.detectAndCompute(imgl_gray,None)
kpr, desr = feature_detect.detectAndCompute(imgr_gray,None)


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
    if m.distance <= 0.7*n.distance:
        ptsr.append(kpr[m.trainIdx].pt)
        ptsl.append(kpl[m.queryIdx].pt)
        good_matches.append([m])

ptsl = np.float32(ptsl)
ptsr = np.float32(ptsr)

matched_image = cv.drawMatchesKnn(imgl, kpl, imgr, kpr, good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow('img',matched_image)
cv.waitKey(0)
#find homography
#then use those keypoints



##calculate the fundamental matrix using the 8 point algorithm
F, mask = cv.findFundamentalMat(ptsl, ptsr, cv.FM_8POINT)
#F, mask = cv.findFundamentalMat(ptsl, ptsr, cv.FM_RANSAC,0.1,0.99)
print('Fundamental Matrix')
print(F)
print('Rank of F')
print(np.linalg.matrix_rank(F))

print('Epipolar Constraint')
ptsl = ptsl[mask.ravel()==1]
ptsr = ptsr[mask.ravel()==1]
#verify  qr^T * F * ql = 0
results = []
for i in range(len(ptsl)):
    #find points
    ptl = ptsl[i]
    ptr = ptsr[i]

    #compute math
    ql = np.array([ptl[0],ptl[1],1.0])
    qr = np.array([ptl[0],ptl[1],1.0])

    x = np.matmul(ql,F)
    x2 = np.matmul(x, np.transpose(qr))
    results.append(x2)

print('Min')
print(np.min(results))
print('Max')
print(np.max(results))
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
print(U)
print(S)
print(VT)

#Tx = [u1 x u2]x
#T is also the third column of the U matrix
cp = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
fm = np.matmul(U,cp)
Tx = np.matmul(fm, np.transpose(U))
T = U[:,2]
T = np.reshape(T,(3,1))
print('Translation Vector')
print(Tx)
print(T)

#R = U[RM]V^T
#two different matricies to represent the two different solutions for this problem
rm = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
rm2 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
fm = np.matmul(U,rm)
fm2 = np.matmul(U,rm)
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
print(P0)
P1_1 = np.append(R,T,1)
P1_2 = np.append(R2,T,1)
P1_3 = np.append(R,-T,1)
P1_4 = np.append(R2,-T,1)

##estimate the reprojection error for both cameras

##triangulate the 3d Points using the linear least square triangulation technique
test_point0 = ptsl[20]
test_point1 = ptsr[20]
my_point = LinearLSTriangulationSVD(test_point0,P0,test_point1,P1_1)
print(my_point)
my_point = LinearLSTriangulationSVD(test_point0,P0,test_point1,P1_2)
print(my_point)
my_point = LinearLSTriangulationSVD(test_point0,P0,test_point1,P1_3)
print(my_point)
my_point = LinearLSTriangulationSVD(test_point0,P0,test_point1,P1_4)
print(my_point)

#pointcloud structure is x,y,z,rgbvalue
#point_cloud = get_pointcloud(ptsl,imgl,ptsr,imgr,P0,P1)

##save to PCD file with features' dominant color
SavePCDToFile(my_point)
#http://www.open3d.org/docs/release/tutorial/reconstruction_system/make_fragments.html
#http://www.open3d.org/docs/release/tutorial/geometry/file_io.html
#http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html


##visualize 3d point cloud with open3D
#http://www.open3d.org/docs/release/tutorial/visualization/visualization.html
#frontal

#top

#side
