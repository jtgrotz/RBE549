import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#function to iterate through given ks and return accuracy
def test_k_range(knn, test, test_labels, max_k):
    ks = np.arange(1,max_k+1)
    accuracies = []
    #test each k_value
    for curr_k in ks:
        ret, result, neighbors, dist = knn.findNearest(test, k=curr_k)
        # check for accuracy
        matches = result == test_labels
        correct = np.count_nonzero(matches)
        accuracy = correct * 100.0 / result.size
        print(accuracy)
        accuracies.append(accuracy)

    return accuracies, ks


img = cv.imread("digits.png")
img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#split image into 5000 cells
cells = [np.hsplit(row,100) for row in np.vsplit(img_grey, 50)]

#convert to numpy array
x = np.array(cells)

#prepare training data and test data
train = x[:,:50].reshape(-1,400).astype(np.float32)
test = x[:,50:100].reshape(-1,400).astype(np.float32)

#create labels and train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:, np.newaxis]
test_labels = train_labels.copy()

#initiate kNN, train and test with k = 1
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)

max_k_value = 9
#collect k data
accuracy_results, ks = test_k_range(knn, test, test_labels, max_k_value)

#plot
plt.plot(ks,accuracy_results)
plt.xlabel('k')
plt.ylabel('accuracy in %')
plt.title('Accuracy vs knn')
plt.legend(['50/50'])
plt.show()