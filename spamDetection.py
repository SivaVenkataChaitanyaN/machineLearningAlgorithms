#!C:/dev/python/3.12.4/_python/venv/sklearn-env/Scripts/python

import re;
import numpy as np;
import pandas as pd;
import sklearn as sk;
import nltk as tk;
from nltk.stem import PorterStemmer;
from nltk.tokenize import word_tokenize;


def getVocabularyList():
    textWordsContent = open('C:/dev/data/ex6/vocab.txt');
    textWordsCnt = textWordsContent.read();
    textWords = textWordsCnt.split("\n");
    vocabList = []
    for i in range(len(textWords)):
        textWord = textWords[i].split("\t");
        vocabList.append(textWord[1]);
    
    return vocabList;

def processEmail(emailContents):
    patterns = [r'<[^<>]>', r'[0-9]+', r"(http|https)://[^\s]*", r"[^\s]+@[^\s]+", r'[$]+'];
    
    replacer = [' ', 'number', 'httpaddr', 'emailaddr', 'dollar'];
    
    for i in range(len(patterns)):
        pattern = patterns[i];
        replace = replacer[i];
        emailContents = re.sub(pattern, replace, emailContents);
    
    print(emailContents)
    
    # wordsList = re.split(r'[ @$/#.-:&*+=[]?!(){},\'\'">_<;%]', emailContents);
    
    # print(wordsList);
    
    ps = PorterStemmer()
    
    wordsList = word_tokenize(emailContents);
    
    for i in range(len(wordsList)): 
        wordsList[i] = re.sub('[^a-zA-Z0-9]', '', wordsList[i]);
    
    stemmedWordsList = []
    
    for w in wordsList:
        if w != '' and len(w) > 1:
            # print(w, " : ",ps.stem(w))
            stemmedWordsList.append(ps.stem(w));
        
    vocabList = getVocabularyList()
    
    stemmedWords = np.reshape(np.array(stemmedWordsList), (len(stemmedWordsList), -1));
    
    # vocabWords = np.reshape(np.array(vocabList), (len(vocabList), -1));
    
    # enc = sk.preprocessing.OneHotEncoder(categories=[vocabList]);
    
    # print(enc.fit(stemmedWords));
    
    # print(enc.transform(stemmedWords).toarray())
    
    indexList = [];
    print(stemmedWordsList)
    
    for word in stemmedWordsList:
        try:
            # print(word, ": ", vocabList.index(word));
            indexList.append(vocabList.index(word));
        except Exception as e:
            continue;
        
    
    return indexList;

##Prepare the Model on given data

dfX = pd.read_csv("C:/dev/data/ex6/spamTrainX.csv", header=None);

dfY = pd.read_csv("C:/dev/data/ex6/spamTrainY.csv", header=None);

svm = sk.svm.SVC(C=0.1, kernel='linear');

svm.fit(dfX, dfY);

pY = svm.predict(dfX);

pY = pY.ravel();

npY = dfY.to_numpy().ravel();

print("Training Data Accuracy percentage is : ", np.sum(pY == npY)/np.shape(pY)[0]);

dfXtest = pd.read_csv("C:/dev/data/ex6/spamTestX.csv", header=None);

dfYtest = pd.read_csv("C:/dev/data/ex6/spamTestY.csv", header=None);

pYTest = svm.predict(dfXtest);

npY = dfYtest.to_numpy().ravel();

pYTest = pYTest.ravel();

print("Test Data Accuracy percentage is : ", np.sum(pYTest == npY)/np.shape(pYTest)[0]);

topIndexList = np.argpartition(svm.coef_, -15)[-15:];

print(topIndexList);

vocabList = getVocabularyList()

print("Top Weighted Words are :")

# for i in range(len(topIndexList[0])) :
    # print(i, ". ",vocabList[topIndexList[0][i]])
    
##Test on a sample Email

try:
    sTextEmail = open('C:/dev/data/ex6/emailSample1.txt');
except Exception as e:
    print("File Not found", e);


emailContents = sTextEmail.read();

print(emailContents);

sTextEmail.close();


indexWordsList = processEmail(emailContents);

X = np.zeros((1899, ));

print(sorted(indexWordsList));
print(len(indexWordsList));

ai = [];

for i in indexWordsList:
    ai.append(i);
    X[i] = 1;
    
print(ai == indexWordsList, ' sumx:', np.sum(X), ' shapex:', np.shape(X), ' ');

#print(np.sum(X));

#print(np.shape(X));

print("Given email is spam or not : ",svm.predict([X]));