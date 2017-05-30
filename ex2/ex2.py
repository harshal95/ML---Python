import numpy as np
import math
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
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
    plt.xlabel('exam 0 score')
    plt.ylabel('exam 1 score')
    plt.legend([leg1,leg2],['Admitted','Not admitted'])
    plt.draw()

def sigmoid(h_x):
    return 1 / (1 + np.exp(-h_x))

def predict(theta,x):

    p = sigmoid(np.dot(theta,x))
    p[p>=0.5] = int(1)
    p[p<0.5] = int(0)
    return p

def plotDecisionBoundary(theta,x,y):

    plt.scatter(x[1],np.multiply(np.dot(theta,x),100),marker='*',color='g')
    plt.show()

def costFun(theta,x,y):
    h_x = np.dot(theta,x)
    h_x = sigmoid(h_x)
    case1 = np.multiply(np.multiply(-1,y),[np.log(np.abs(t)) for t in h_x])
    h_x_inv = [1 - t for t in h_x]
    case2 = np.multiply([1-t for t in y],[np.log(np.abs(h)) for h in h_x_inv])
    summed = np.subtract(case1,case2)
    cost = np.sum(summed)/y.size
    return cost

def gradient(theta,x,y):
    h_x = np.dot(theta, x)
    h_x = sigmoid(h_x)
    grad = np.zeros(theta.size)
    for i in range(theta.size):
        grad[i] = np.sum(np.multiply(np.subtract(h_x,y),x[i]))/y.size
    return grad

if __name__ == '__main__':
    df = pd.read_csv('ex2data1.txt',header=-1)
    plot(df)
    df_len = len(df)
    x1 = np.reshape(df[0].values,(df_len,1))
    x2 = np.reshape(df[1].values,(df_len,1))
    y = np.reshape(df[2].values,(df_len,1))
    x0 = np.ones((df_len,1))
    x_temp = np.concatenate([x1,x2],axis=1)
    x = np.concatenate([x0,x_temp],axis=1)
    x = x.transpose()
    y = y.transpose()
    theta = np.zeros((1,x.shape[0]))
    cost= costFun(theta,x,y)
    grad = gradient(theta,x,y)
    # theta[0][0] = -24
    # theta[0][1] = 0.2
    # theta[0][2] = 0.2
    # cost = costFun(x, y, theta)
    # grad = gradient(x, y, theta)
    result = opt.fmin_tnc(func=costFun,x0=theta,fprime=gradient,args=(x,y))
    theta = result[0]
    min_cost = costFun(theta,x,y)
    #plotDecisionBoundary(theta,x,y)
    prob = sigmoid(np.dot(theta,np.array([1,45,85])))
    pred = predict(theta,x)
    pred= pred.astype(int)
    accuracy = (pred==y)
    ans = (float(np.count_nonzero(accuracy))/df_len)*100
    print
    print