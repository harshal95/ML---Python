import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h



if __name__ == '__main__':
    data = loadmat('ex3data1.mat')
    X = data['X']
    y = data['y']
    X.shape, y.shape
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)
    y_onehot.shape
    thetas = loadmat('ex3weights.mat')
    theta1 = thetas['Theta1']
    theta2 = thetas['Theta2']
    X = np.matrix(X)
    y = np.matrix(y)
    theta1 = np.matrix(theta1)
    theta2 = np.matrix(theta2)
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    y_pred = np.array(np.argmax(h, axis=1) + 1)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print 'accuracy = {0}%'.format(accuracy * 100)
    print

