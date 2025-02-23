#!C:/dev/python/3.12.4/_python/venv/sklearn-env/Scripts/python

import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np;
import sklearn as sk;
# import tensorflow as tf;
# import tensorflow_probability as tfp;
# import tensorflow_transform as tft;

dfX = pd.read_csv("C:/dev/data/ex7/PCA_X.csv", header=None);

figure = plt.figure(facecolor='lightblue');

axes = figure.add_subplot();

npX = dfX.to_numpy();

meanX = np.mean(npX, axis=0);

stdX = np.std(npX, axis=0);

npX = npX - meanX;

npX = npX / stdX;

axes.plot(npX[:,0], npX[:, 1], "o", color="blue")

# print(tft.pca(dfX))

pca = sk.decomposition.PCA(n_components=1)

XPCA = pca.fit_transform(dfX);

Xrec = XPCA * pca.components_

axes.plot(Xrec[:, 0], Xrec[:, 1], "o", color="red")

dfX = pd.read_csv("C:/dev/data/ex7/faces.csv", header=None);

npX = dfX.to_numpy();

meanX = np.mean(npX, axis=0);

stdX = np.std(npX, axis=0);

npX = npX - meanX;

npX = npX / stdX;

pca = sk.decomposition.PCA(n_components=100)

XPCA = pca.fit_transform(dfX);

Xrec = np.matmul(XPCA, pca.components_);


def displayFaces(X, axes) :
    [m, n] = np.shape(X);
    
    width = (int) (np.round(np.sqrt(n)));
    
    height = (int) (n/width);
    
    display_rows = (int) (np.floor(np.sqrt(m)));
    
    display_cols = (int) (np.ceil(m / display_rows));
    
    pad = 1;
    
    data = -np.ones(((int)(pad + display_rows*(pad + height)), (int)(pad + display_cols*(pad+width))));
    
    for j in range(display_rows):
        for i in range(display_cols):
            begin = (int)(pad + j * (height + pad))
            end = (int)(pad + i * (width + pad))
            cur = i+j;
            data[begin:(begin+height), end:(end+width)] = np.reshape(X[cur, :], (height, width))
    
    axes.imshow(data, cmap='Greys_r');


figure, axes = plt.subplots(1, 2);

displayFaces(npX[0:100, :], axes[0]);

displayFaces(Xrec[0:100, :], axes[1]);

plt.show();

