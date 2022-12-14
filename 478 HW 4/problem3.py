import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as random

class KMeans(object):
    # K is the K in KMeans
    # useKMeansPP is a boolean. If True, you should initialize using KMeans++
    def __init__(self, K, useKMeansPP):
        self.K = K
        self.useKMeansPP = useKMeansPP

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        K = self.K
        N = len(X)

        # Initialize Centroids
        centroids = []
        for k in range(K):
            centroids.append(random.sample(pics, 1)[0])

        objectives = []
        labels = np.zeros(N)

        count = 1
        while count != 0:
            count = 0

            for i in range(N):

                dist = []
                for j in range(K):
                    dist.append((np.linalg.norm(X[i]-centroids[j]))**2)
                closest = np.argmin(dist)

                if labels[i] != closest:
                    labels[i] = closest
                    count += 1

            for j in range(K):
                count = 0
                values = np.zeros([28,28])
                for i in range(N):
                    if labels[i] == j:
                        count += 1
                        values += X[i]
                centroids[j] = values/count
            
            objective = 0
            for n in range(N):
                for k in range(K):
                    if labels[n] == k:
                        objective += (np.linalg.norm(X[n]-centroids[k]))**2
            
            objectives.append(objective)
        
        self.X = X
        self.labels = labels
        self.centroids = centroids
        self.objectives = objectives
        
    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        centroids = self.centroids
        self.create_image_from_array(centroids)
        pass

    # This should return the arrays for D images from each cluster that are representative of the clusters.
    def get_representative_images(self, D):
        pass

    # img_array should be a 2D (square) numpy array.
    # Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
    # However, we do ask that any images in your writeup be grayscale images, just as in this example.
    def create_image_from_array(self, img_array):
        plt.figure()
        plt.imshow(img_array, cmap='Greys_r')
        plt.show()
        return

# This line loads the images for you. Don't change it! 
pics = np.load("images.npy", allow_pickle=False)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.
K = 10
KMeansClassifier = KMeans(K=10, useKMeansPP=False)
KMeansClassifier.fit(pics)
KMeansClassifier.create_image_from_array(pics[1])




