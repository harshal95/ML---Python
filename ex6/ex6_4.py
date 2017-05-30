import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

if __name__ == '__main__':
    spam_train = loadmat('spamTrain.mat')
    spam_test = loadmat('spamTest.mat')
    print spam_train
    X = spam_train['X']
    Xtest = spam_test['Xtest']
    y = spam_train['y'].ravel()
    ytest = spam_test['ytest'].ravel()
    svc = svm.SVC()
    svc.fit(X, y)
    print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))  