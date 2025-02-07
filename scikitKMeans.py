#!C:/dev/python/3.12.4/_python/venv/sklearn-env/Scripts/python
import pandas as pd;
import sklearn as sk;
import numpy as np;
import matplotlib.pyplot as plt
import matplotlib.image as img


dfX = pd.read_csv("C:/dev/data/ex7/ex7KMeans.csv", header=None);

npX = dfX.to_numpy();

K = 3;

centroids = np.array([[3, 3],[6, 2], [8, 5]]);

def findClosestCentroids(X, centroids) :
    finalCentroids = np.zeros((np.shape(X)[0], 1));
    
    for i in range(np.shape(X)[0]):
        lowDist = 0;
        nearestCentroid = 0;
        for j in range(K):
            calculatedDist = np.sum(np.pow(np.subtract(X[i, :], centroids[j, :]), 2))
            
            if (j == 0 or calculatedDist < lowDist):
                lowDist = calculatedDist
                nearestCentroid = j;
        
        finalCentroids[i] = nearestCentroid;
    return finalCentroids;

closestCentroids = findClosestCentroids(npX, centroids);

closestCentroids = closestCentroids.astype(int);

print(closestCentroids.dtype);

def computeClosestCentroids(X, closestCentroids):
    newCentroids = np.zeros((K, np.shape(X)[1]));
    
    for i in range(np.shape(X)[0]):
        newCentroids[closestCentroids[i], :] += X[i, :]
    
    for i in range(K):
        count = np.sum(closestCentroids == i)
        newCentroids[i, :] /= count;
    
    return newCentroids
 

print(computeClosestCentroids(npX, closestCentroids))

kmeans = sk.cluster.KMeans(n_clusters=3, init=np.array([[3, 3], [6, 2], [8, 5]]), max_iter=300).fit(npX);

figure, axes = plt.subplots()

for i in range(np.shape(npX)[0]):
    if(kmeans.labels_[i] == 0):
        axes.plot(npX[i, 0], npX[i, 1], 'yo');
    elif (kmeans.labels_[i] == 1):
        axes.plot(npX[i, 0], npX[i, 1], 'ro');
    else:
        axes.plot(npX[i, 0], npX[i, 1], 'go');

axes.plot(kmeans.cluster_centers_[0, 0], kmeans.cluster_centers_[0, 1], 'bx', markersize=20, markeredgewidth=5);

axes.plot(kmeans.cluster_centers_[1, 0], kmeans.cluster_centers_[1, 1], 'bx', markersize=20, markeredgewidth=5);

axes.plot(kmeans.cluster_centers_[2, 0], kmeans.cluster_centers_[2, 1], 'bx', markersize=20, markeredgewidth=5);

# plt.show();

imgData = img.imread("C:/dev/data/ex7/bird_small.png", "png");

X = np.reshape(imgData, (16384, 3))

print(np.shape(X));

figure, axes = plt.subplots()

axes.imshow(imgData);

K = 16;

imgKmeans = sk.cluster.KMeans(16, init='random', max_iter=10).fit(X);

print(imgKmeans.cluster_centers_);

print(imgKmeans.n_features_in_);

for i in range(np.shape(X)[0]):
    X[i, :] = imgKmeans.cluster_centers_[imgKmeans.labels_[i], :]

imgData2 = np.reshape(X, (128, 128, 3));

figure, axes = plt.subplots()

axes.imshow(imgData2);

plt.show();


