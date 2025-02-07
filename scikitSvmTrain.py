#!C:/dev/python/3.12.4/_python/venv/sklearn-env/Scripts/python

import pandas as pd;
import sklearn as sk;
import numpy as np;
import matplotlib.pyplot as plt


dfX = pd.read_csv("C:/dev/data/ex6/linearKernelX.csv", header=None);

dfY = pd.read_csv("C:/dev/data/ex6/linearKernelY.csv", header=None);

npX = dfX.to_numpy();

npY = dfY.to_numpy();

ones = dfY.index[dfY[0] == 1];

zeros = dfY.index[dfY[0] == 0];

figure = plt.figure(facecolor='lightblue');

axes = figure.add_subplot();

onesX = dfX.drop(zeros);

zerosX = dfX.drop(ones);

axes.plot(onesX[0], onesX[1], '+');

axes.plot(zerosX[0], zerosX[1], 'o');

dfY = dfY.values.reshape(dfY.shape[0]);

svm = sk.svm.SVC(kernel='linear');

svm.fit(dfX, dfY);

npX = dfX.to_numpy();

plotX = np.linspace(np.min(npX[:, 0]), np.max(npX[:, 0]), 100);

plotY = -((svm.coef_[0][0]*plotX)/svm.coef_[0][1] + svm.intercept_[0]/svm.coef_[0][1]);

axes.plot(plotX, plotY);

# plt.show();

dfX = pd.read_csv("C:/dev/data/ex6/guassianKernelX.csv", header=None);

dfY = pd.read_csv("C:/dev/data/ex6/guassianKernelY.csv", header=None);

npX = dfX.to_numpy();

npX2 = np.sum(np.pow(npX, 2), 1);

npTemp = np.transpose(npX2) - (2 *np.matmul(npX, np.transpose(npX)));

npK = npX2 + npTemp;

npK = np.pow(np.exp(-50), npK)

# print(npK);

svm = sk.svm.SVC();

# svm = sk.svm.SVC(kernel='precomputed');

ones = dfY.index[dfY[0] == 1];

zeros = dfY.index[dfY[0] == 0];

onesX = dfX.drop(zeros);

zerosX = dfX.drop(ones);

fig, axes = plt.subplots()

axes.plot(onesX[0], onesX[1], '+');

axes.plot(zerosX[0], zerosX[1], 'o');

dfY = dfY.values.reshape(dfY.shape[0])

svm.fit(dfX, dfY);

# svm.fit(npK, dfY);

npX = dfX.to_numpy();

plotX1 = np.linspace(np.min(npX[:, 0]), np.max(npX[:, 0]), 863)

plotX2 = np.linspace(np.min(npX[:, 1]), np.max(npX[:, 1]), 863)

x, y = np.meshgrid(plotX1, plotX2, indexing='ij');

x = np.transpose(x)

y = np.transpose(y)

z = np.zeros((np.shape(x)[0], np.shape(x)[1]))

# print(svm.predict([x[0][0], y[0][0]]))
for i in range(0, np.shape(x)[0]):
    for j in range(0, np.shape(y)[0]):
        z[i, j] = svm.predict([[x[i][j], y[i][j]]])[0]


contourX = pd.read_csv("C:/dev/data/ex6/contourX.csv", header=None).to_numpy();
contourY = pd.read_csv("C:/dev/data/ex6/contourY.csv", header=None).to_numpy();
contourZ = pd.read_csv("C:/dev/data/ex6/contourZ.csv", header=None).to_numpy();

# for i in range(0, 100): 
    # if(x[0][i] != contourX[0][i]):
        # print("i: 0", " j: ", i);
        # print("x : ", x[0][i], " contour: ", contourX[0][i]);
            
# for i in range(0, 100):
    # for j in range(0, 100):
        # if(z[i][j] != contourZ[i][j]):
            # print("i: ", i," j: ", j);
            # print("z : ", z[i][j], " contour: ", contourZ[i][j]);
            

axes.contour(x, y, z);

# plt.show();

dfX = pd.read_csv("C:/dev/data/ex6/guassianDataset3X.csv", header=None);

dfY = pd.read_csv("C:/dev/data/ex6/guassianDataset3Y.csv", header=None);

svm = sk.svm.SVC();

fig, axes = plt.subplots()

ones = dfY.index[dfY[0] == 1];

zeros = dfY.index[dfY[0] == 0];

onesX = dfX.drop(zeros);

zerosX = dfX.drop(ones);

axes.plot(onesX[0], onesX[1], '+');

axes.plot(zerosX[0], zerosX[1], 'o');

dfY = dfY.values.reshape(dfY.shape[0])

svm.fit(dfX, dfY);

npX = dfX.to_numpy();

plotX1 = np.linspace(np.min(npX[:, 0]), np.max(npX[:, 0]), 100)

plotX2 = np.linspace(np.min(npX[:, 1]), np.max(npX[:, 1]), 100)

x, y = np.meshgrid(plotX1, plotX2, indexing='ij');

x = np.transpose(x)

y = np.transpose(y)

z = np.zeros((np.shape(x)[0], np.shape(x)[1]))

# print(svm.predict([x[0][0], y[0][0]]))
for i in range(0, np.shape(x)[0]):
    for j in range(0, np.shape(y)[0]):
        z[i, j] = svm.predict([[x[i][j], y[i][j]]])[0]
        
axes.contour(x, y, z);

plt.show();