import numpy as np
import scipy.io
import sklearn
import re
import matplotlib.pyplot as plt
from nltk import PorterStemmer

def processEmail(Email): # you kinda give up when regex entered the chat 
    with open('vocab.txt', 'r') as vocab:
        vocabList = {}
        for line in vocab.readlines():
            i, word = line.split() #split empty splits on the \ 
            vocabList[word] = int(i)
    Email=Email.lower()
    Email = re.sub('<[^<>]+>', ' ', Email)
    Email = re.sub('[0-9]+', 'number', Email)
    Email = re.sub('(http|https)://[^\s]*', 'httpaddr', Email)
    Email = re.sub('[^\s]+@[^\s]+', 'emailaddr', Email)
    Email = re.sub('[$]+', 'dollar', Email)
    Email = re.split(r'[@$/#.-:&\*\+=\[\]?!(){},\'\'\">_<;%\s\n\r\t]+', Email)
    word_indices=[]
    l=0
    for token in Email:
        token = re.sub('[^a-zA-Z0-9]', '', token)
        token = PorterStemmer().stem(token.strip())
        if len(token) < 1:
            continue
        idx = vocabList[token] if token in vocabList else 0
        if idx > 0:word_indices.append(idx)
        if l + len(token) + 1 > 78:l = 0
        l = l + len(token) + 1
    return word_indices

def feature(E):
    f=np.zeros((words_in_dic))
    f[E]=1
    return f

    
    
words_in_dic=1899
Data=open('emailSample1.txt', 'r').read()
Email=processEmail(Data)
features=feature(Email)


c=.1
Data=scipy.io.loadmat('spamTrain.mat')
X,y=Data['X'],np.matrix(Data['y']).A1
model = sklearn.svm.SVC(C=c, kernel="linear", tol=1e-3).fit(X,y)#

p= model.predict(X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(p == y) * 100))

DataT=scipy.io.loadmat('spamTest.mat')
X,y=DataT['Xtest'],np.matrix(DataT['ytest']).A1
p= model.predict(X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(p == y) * 100))
# part 5 and 6 to be continued 



