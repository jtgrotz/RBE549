import cv2 as cv
import numpy as np

SZ=20
bin_num = 16

affine_flag = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR

#function for deskewing image
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1,skew,-0.5*SZ*skew], [0,1,0]])
    img = cv.warpAffine(img,M,(SZ,SZ), flags=affine_flag)
    return img

#function for computing histogram of gradients
def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_num * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_num) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist

#load image

digits = cv.imread('digits.png', cv.IMREAD_GRAYSCALE)

#shuffle data points

#split into cells
cells = [np.hsplit(row,100) for row in np.vsplit(digits,50)]

#set train and test
train_cells = [i[:50] for i in cells]
test_cells = [i[50:] for i in cells]

#deskew, form hog vectors, and set as train data, and generate responses
deskewed = [list(map(deskew,row)) for row in train_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
trainData = np.float32(hogdata).reshape(-1,64)
responses = np.repeat(np.arange(10),250)[:,np.newaxis]

#create SVM with right parameters
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

svm.train(trainData,cv.ml.ROW_SAMPLE, responses)

#deskew, form hog vectors, and set as test data, and generate responses
deskewed = [list(map(deskew,row)) for row in test_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
testData = np.float32(hogdata).reshape(-1,bin_num*4)
result = svm.predict(testData)[1]


#calculate response rate
mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)