from ucimlrepo import fetch_ucirepo
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#pip install ucimlrepo
# if you havent done that



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
        accuracies.append(accuracy)

    return accuracies, ks

#function to iterate over train percents, and test accuracy for ks.
#train_percents are specified as 10 for 10%, which would indicate 10% train, 90% test
def test_split_range(data,labels,train_percents,max_k):
    split_accuracies = []
    ks = 0
    for split in train_percents:
        print(split)
        data_size = data.shape[0]
        #find new splits for data train and test
        train_indices = np.int16(split/100 * data_size)
        test_indicies = data_size-train_indices
        train = data.values[:train_indices,:].reshape(-1,16).astype(np.float32)
        test = data.values[train_indices:data_size,:].reshape(-1,16).astype(np.float32)

        #make new labels given splits
        train_labels = labels.values[:train_indices]
        test_labels = labels.values[train_indices:data_size]

        #convert to numerical values in the right format for matching
        train_labels = np.array([[ord(ch[0])] for ch in train_labels])
        test_labels = np.array([[ord(ch[0])] for ch in test_labels])

        # initiate kNN
        knn = cv.ml.KNearest_create()
        knn.train(train, cv.ml.ROW_SAMPLE, train_labels)

        #test accuracy over k for current split
        accuracy_results, ks = test_k_range(knn, test, test_labels, max_k)
        split_accuracies.append(accuracy_results)

    return split_accuracies, ks



# fetch dataset
letter_recognition = fetch_ucirepo(id=59)

# data
features = letter_recognition.data.features
targets = letter_recognition.data.targets

#create training splits
training_splits = [10,20,30,40,50,60,70,80,90]
max_k_range = 9
results, ks = test_split_range(features,targets,training_splits,max_k_range)

#plotting of results
print(results)
for i in range(len(results)):
    accuracies = results[i]
    plt.plot(ks,accuracies)

plt.xlabel('K')
plt.ylabel('Accuracy in %')
plt.title('Accuracy vs K for different Splits (Train/Test) Alphabet')
plt.legend(["10/90", "20/80", "30/70","40/60","50/50","60/40","70/30","80/10","90/10"])
plt.show()