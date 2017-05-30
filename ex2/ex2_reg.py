import numpy as np
import math
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def plot(df):
    y = df[2]
    pos = y==1
    neg = y==0
    pos_df = df[pos]
    neg_df = df[neg]
    x0_pos = pos_df[0]
    x1_pos = pos_df[1]
    x0_neg = neg_df[0]
    x1_neg = neg_df[1]
    leg1 = plt.scatter(x0_pos,x1_pos,marker='x',color='r')
    leg2 = plt.scatter(x0_neg,x1_neg,marker='+',color='b')
    plt.xlabel('Microchip test 1')
    plt.ylabel('Microchip test 2')
    plt.legend([leg1,leg2],['y = 1','y = 0'])
    plt.show()

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg

def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])

    return grad
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

if __name__ == '__main__':
    df = pd.read_csv('ex2data2.txt',header=None,names = ['Test 1', 'Test 2', 'Accepted'])
    #plot(df)
    degree = 5
    x1 = df['Test 1']
    x2 = df['Test 2']
    df.insert(3,'Ones',1)
    for i in range(1,degree):
        for j in range(0,i):
            df['F' + str(i) + str(j)] = np.power(x1,i-j) * np.power(x2,j)
    df.drop('Test 1',axis=1,inplace=True)
    df.drop('Test 2', axis=1, inplace=True)
    cols = df.shape[1]
    X2 = df.iloc[:, 1:cols]
    y2 = df.iloc[:, 0:1]
    X2 = np.array(X2.values)
    y2 = np.array(y2.values)
    theta2 = np.zeros(11)
    learningRate = 1
    cost = costReg(theta2, X2, y2, learningRate)
    grad = gradientReg(theta2,X2,y2,learningRate)

    result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))

    theta_min = np.matrix(result2[0])
    predictions = predict(theta_min, X2)

    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print 'accuracy = {0}%'.format(accuracy)
    print