import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

# import images

book = cv.imread('book.jpg',cv.IMREAD_GRAYSCALE)
#resizing image to fit on screen better
book = cv.resize(book, None, fx=0.7, fy=0.7)

table = cv.imread('table.jpg',cv.IMREAD_GRAYSCALE)
#resizing image to fit on screen better
table = cv.resize(table, None, fx=0.7, fy=0.7)

#create SIFT detector and find sift keypoints
sift = cv.SIFT_create()

tic = time.perf_counter()
book_sift_kp, book_sift_des = sift.detectAndCompute(book,None)
table_sift_kp, table_sift_des = sift.detectAndCompute(table,None)
toc = time.perf_counter()
print(f"SIFT detection took {toc-tic:0.8f} seconds")

#creater SURF detector and find surf keypoints
surf = cv.xfeatures2d.SURF_create(400)

tic = time.perf_counter()
book_surf_kp, book_surf_des = surf.detectAndCompute(book,None)
table_surf_kp, table_surf_des = surf.detectAndCompute(table,None)
toc = time.perf_counter()
print(f"SURF detection took {toc-tic:0.8f} seconds")

#set up brute force, use base setup to all for knn searching
bf = cv.BFMatcher()

#set up FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)


#Combo 1 SIFT and Brute force
tic = time.perf_counter()
match1 = bf.knnMatch(book_sift_des, table_sift_des, k=2)
match1_good = []
#ratio test
for m,n in match1:
    if m.distance < 0.7*n.distance:
        match1_good.append([m])
toc = time.perf_counter()
print(f"match 1 took {toc-tic:0.8f} seconds")
#depict matches and show image
im_match1 = cv.drawMatchesKnn(book, book_sift_kp, table, table_sift_kp, match1_good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow('img',im_match1)
cv.waitKey(0)

#Combo 2 SIFT and FLANN
tic = time.perf_counter()
match2 = flann.knnMatch(book_sift_des,table_sift_des,k=2)
match2_mask = [[0,0] for i in range(len(match2))]
#ratio test
for i,(m,n) in enumerate(match2):
    if m.distance < 0.7*n.distance:
        match2_mask[i]=[1,0]
toc = time.perf_counter()
print(f"match 2 took {toc-tic:0.8f} seconds")
#depict matches and show image
im_match2 = cv.drawMatchesKnn(book, book_sift_kp, table, table_sift_kp, match2, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchesMask=match2_mask)
cv.imshow('img',im_match2)
cv.waitKey(0)


#Combo 3 SURF and Brute force
tic = time.perf_counter()
match3 = bf.knnMatch(book_surf_des, table_surf_des, k=2)
match3_good = []
#ratio test
for m,n in match1:
    if m.distance < 0.7*n.distance:
        match3_good.append([m])
toc = time.perf_counter()
print(f"match 3 took {toc-tic:0.8f} seconds")
#depict matches and show image
im_match3 = cv.drawMatchesKnn(book, book_surf_kp, table, table_surf_kp,match3_good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow('img', im_match3)
cv.waitKey(0)


#Combo 3 SURF and FLANN
tic = time.perf_counter()
match4 = flann.knnMatch(book_surf_des,table_surf_des,k=2)
match4_mask = [[0,0] for i in range(len(match4))]
#ratio test
for i,(m,n) in enumerate(match4):
    if m.distance < 0.7*n.distance:
        match4_mask[i]=[1,0]

toc = time.perf_counter()
print(f"match 4 took {toc-tic:0.8f} seconds")

#depict matches and show image
im_match4 = cv.drawMatchesKnn(book, book_surf_kp, table, table_surf_kp, match4, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchesMask=match4_mask)
cv.imshow('img',im_match4)
cv.waitKey(0)

