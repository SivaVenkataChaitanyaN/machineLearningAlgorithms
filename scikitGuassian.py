#!C:/dev/python/3.12.4/_python/venv/sklearn-env/Scripts/python

import pandas as pd;
import numpy as np;
import sklearn as sk;
import matplotlib.pyplot as plt;

dfX = pd.read_csv("C:/dev/data/ex8/MultiVariantX.csv", header=None);

npX = dfX.to_numpy();

figure, axes = plt.subplots();

axes.plot(npX[:, 0], npX[:, 1], 'bo');

# plt.show();

model = sk.mixture.GaussianMixture(n_components=3).fit(npX);

x1, x2 = np.meshgrid(np.arange(0, 36, 0.5), np.arange(0, 36, 0.5), indexing='ij')

plotX = np.transpose(np.array([x1.ravel(), x2.ravel()]))

Z = model.score_samples(plotX);

Z = np.reshape(Z, (72, 72));

axes.contour(x1, x2, Z);

# plt.show();

dfXval = pd.read_csv("C:/dev/data/ex8/MultiVariantXval.csv", header=None);

dfYval = pd.read_csv("C:/dev/data/ex8/MultiVariantYval.csv", header=None);

# valModel = sk.mixture.GaussianMixture(n_components=3).fit(dfXval);

probabilities = model.score_samples(dfXval);

# probabilities = -probabilities;

# print(probabilities);

# print(np.shape(probabilities));

# print(dfYval);



def calculateEpilon(dfYval, valProbabilities):
    optimalF1 = 0;
    
    optimalEpsilon = 0;
    
    stepSize = (np.max(valProbabilities) - np.min(valProbabilities))/1000;
    
    print("stepSize: ",stepSize);
    
    print("min: ",np.min(valProbabilities), " max: ", np.max(valProbabilities));
    
    npEplison = np.arange(np.min(valProbabilities), np.max(valProbabilities), stepSize);
    
    for i in npEplison:
        valPrediction = (valProbabilities < i).astype(int);
        
        nwTruePositive = np.sum((valPrediction == 1) & (dfYval == 1));
        
        nwTrueNegative = np.sum((valPrediction == 0) & (dfYval == 0));
        
        nwFalsePositive = np.sum((valPrediction == 1) & (dfYval == 0));
        
        nwFalseNegative = np.sum((valPrediction == 0) & (dfYval == 1));
        
        nfPrecision = 0;

        nfRecall = 0;
        
        nfF1 = 0;

        if(nwTruePositive != 0 or nwFalsePositive != 0):
            nfPrecision = nwTruePositive/(nwTruePositive + nwFalsePositive);
        
        
        if(nwTruePositive != 0 or nwFalseNegative != 0):
           nfRecall = nwTruePositive/(nwTruePositive + nwFalseNegative); 

        if(nfPrecision != 0 or nfRecall != 0):
           nfF1 = (2 * nfPrecision * nfRecall)/(nfPrecision + nfRecall);
        
        
        if(nfF1 > optimalF1):
            optimalF1 = nfF1;
            
            optimalEpsilon = i;
    
    return (optimalF1, optimalEpsilon);

(F1, Epsilon) = calculateEpilon(dfYval.to_numpy(), probabilities);  

print(Epsilon);

initialProb = model.score_samples(npX)

outliers = initialProb < Epsilon

print(outliers);

indices = np.where(outliers == True);

axes.plot(npX[indices, 0], npX[indices, 1], "rx", linewidth=2, markersize=10);

plt.show();