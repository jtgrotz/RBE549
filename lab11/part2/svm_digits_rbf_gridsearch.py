import cv2 as cv
import numpy as np
import random
from matplotlib import pyplot as plt

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

def GridSearch(c,gamma, train_data, train_responses, test_data, test_responses):
    index = 0
    accuracy_results = []
    c_setting = []
    gamma_setting = []
    for curr_c in c:
        for curr_g in gamma:
            # create SVM with right parameters
            svm = cv.ml.SVM_create()
            svm.setKernel(cv.ml.SVM_LINEAR)
            svm.setType(cv.ml.SVM_C_SVC)
            svm.setC(curr_c)
            svm.setGamma(curr_g)
            svm.train(train_data, cv.ml.ROW_SAMPLE, train_responses)

            # calculate response rate
            predicted_result = svm.predict(test_data)[1]
            mask = predicted_result == test_responses
            correct = np.count_nonzero(mask)
            accuracy_results.append(correct*100.0/predicted_result.size)
            c_setting.append(curr_c)
            gamma_setting.append(curr_g)
            index += 1
            print(index)
            print(curr_c)
            print(curr_g)
    return accuracy_results, c_setting, gamma_setting





#load image

digits = cv.imread('digits.png', cv.IMREAD_GRAYSCALE)


#split into cells
cells = [np.hsplit(row,100) for row in np.vsplit(digits,50)]

#shuffle data points
#shuffled_cells, shuffled_labels = shuffle(cells)

#set train and test
#train_cells = shuffled_cells[:trainnum]
#test_cells = shuffled_cells[trainnum:]
train_cells = [ i[:50] for i in cells]
test_cells = [ i[50:] for i in cells]

#shuffle

#deskew, form hog vectors, and set as train data, and generate responses
#deskewed = [deskew(i) for i in train_cells]
#hogdata = [hog(i) for i in deskewed]
deskewed = [list(map(deskew,row)) for row in train_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
trainData = np.float32(hogdata).reshape(-1,64)
#responses = shuffled_labels[:trainnum]
responses = np.repeat(np.arange(10),250)[:,np.newaxis]


#deskew, form hog vectors, and set as test data, and generate responses
#deskewed2 = [deskew(i) for i in test_cells]
#hogdata2 = [hog(i) for i in deskewed2]
deskewed = [list(map(deskew,row)) for row in test_cells]
hogdata2 = [list(map(hog,row)) for row in deskewed]
testData = np.float32(hogdata2).reshape(-1,bin_num*4)
#test_responses = shuffled_labels[trainnum:]
test_responses = np.repeat(np.arange(10),250)[:,np.newaxis]

C = [1, 20, 150, 900]
GAMMA = [0.2, 0.4, 0.6, 0.8, 0.9, 1.2, 1.5, 1.7]

accuracy, c_set, g_set = GridSearch(C,GAMMA,trainData,responses,testData,test_responses)
print(accuracy)
print(np.max(accuracy))


#3D scatter plotting
ax = plt.axes(projection='3d')

ax.scatter3D(c_set,g_set,accuracy,c=accuracy)
ax.set_ylabel('Gammas')
ax.set_xlabel('Cs')
ax.set_zlabel('Accuracy in %')
plt.show()



