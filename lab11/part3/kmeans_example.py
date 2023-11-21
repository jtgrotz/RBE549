import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('nature.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
Z = img.reshape((-1,3))
#convert to float 32
Z = np.float32(Z)

#criteria for Kmeans
criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 20, 0.5)

quantized_images = []
Ks = [2,3,5,10,20,40]
subplot_index = 1
for K in Ks:
    print(K)
    #find clusters
    ret,label,center = cv.kmeans(Z,K,None,criteria,20,cv.KMEANS_RANDOM_CENTERS)

    #recreate original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    #plot in subplot
    plt.subplot(2,3,subplot_index)
    plt.imshow(res2)
    plt.title('K='+str(K))
    subplot_index += 1

plt.show()



