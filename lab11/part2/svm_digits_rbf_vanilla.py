import cv2 as cv
import numpy as np
import random

SZ=20
bin_num = 16

trainnum = 4500
testnum = 500

affine_flag = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR

#function for shuffling the test data for better results
#returns shuffled list, and appropriate label
def shuffle(data):
    #create generic labels array
    labels = np.repeat(np.arange(10), 500)[:, np.newaxis]
    new_labels = []
    new_data_list = list()
    #create random nonrepeating sequence
    ran_list = random.sample(range(len(labels)),len(labels))
    for i in ran_list:
        new_labels.append(labels[i])
        #convert to proper dimension for indexing
        d1 = np.int16(np.floor(i/100))
        d2 = np.int16(i-(d1*100))
        new_data_list.append(data[d1][d2])

    new_labels = np.array(new_labels)
    #new_data_list = np.array(new_data_list)
    return new_data_list, new_labels


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


#split into cells
cells = [np.hsplit(row,100) for row in np.vsplit(digits,50)]

#shuffle data points
shuffled_cells, shuffled_labels = shuffle(cells)

#set train and test
train_cells = shuffled_cells[:trainnum]
test_cells = shuffled_cells[trainnum:]

#shuffle

#deskew, form hog vectors, and set as train data, and generate responses
deskewed = [deskew(i) for i in train_cells]
hogdata = [hog(i) for i in deskewed]
trainData = np.float32(hogdata).reshape(-1,64)
responses = shuffled_labels[:trainnum]

#create SVM with right parameters
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_RBF)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(12)
svm.setGamma(0.5)

#svm.train(trainData,cv.ml.ROW_SAMPLE, responses)
svm.trainAuto(trainData,cv.ml.ROW_SAMPLE, responses)

#deskew, form hog vectors, and set as test data, and generate responses
deskewed2 = [deskew(i) for i in test_cells]
hogdata2 = [hog(i) for i in deskewed2]
testData = np.float32(hogdata2).reshape(-1,bin_num*4)
responses = shuffled_labels[trainnum:]
result = svm.predict(testData)[1]


#calculate response rate
print(result[:25])
print(responses[:25])
mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)