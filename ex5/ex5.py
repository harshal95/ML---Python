import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
def plot(X,y):

    plt.scatter(X,y,marker='x',color='b')
    plt.title('data')
    plt.xlabel('Change in water level')
    plt.ylabel('Water flowing out of the dam')
    plt.xlim(-50,40)
    plt.ylim(-5,40)

    plt.draw()
def plotLine(X,theta):
    plt.scatter(X[1], np.dot(theta, X), marker='*', color='r')
    plt.show()

def plotCurve(tcost,vcost):
    plt.scatter(np.arange(tcost.shape[0]),tcost[:,0], marker='x', color='b')
    plt.draw()
    plt.scatter(np.arange(vcost.shape[0]), vcost[:, 0], marker='x', color='g')
    plt.show()

def computeCost(theta,X,y,learningRate):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    inner = np.power(((X*theta.T)-y),2)
    cost = np.sum(inner)/(2*X.shape[0])
    reg = np.sum(np.power(theta[:,1:],2))*learningRate/(2*X.shape[0])
    return cost + reg

def gradient(theta,X,y,learningRate):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    error = X*theta.T - y
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)
    return np.array(grad).ravel()

def learningCurve(data):
    Xt = data['X']
    Xt = np.insert(Xt, 0, values=np.ones(Xt.shape[0]), axis=1)
    yt = data['y']
    m = Xt.shape[0]
    Xv = data['Xval']
    Xv = np.insert(Xv, 0, values=np.ones(Xv.shape[0]), axis=1)
    yv = data['yval']
    theta = np.ones([1,2],dtype=int)
    tcost = np.empty((m,1))
    vcost = np.empty((m,1))
    learningRate = 0.0
    for i in range(m):
        fmin = minimize(fun=computeCost, x0=theta, args=(Xt[0:i+1,], yt[0:i+1,], learningRate), method='TNC', jac=gradient)
        theta = np.array(fmin.x)
        tcost[i,0] = computeCost(theta,Xt[0:i+1,], yt[0:i+1,], learningRate)
        vcost[i,0] = computeCost(theta,Xv, yv, learningRate)
        theta = np.ones([1, 2], dtype=int)
    return tcost,vcost

if __name__=='__main__':
    data = loadmat('ex5data1.mat')
    X = data['X']
    y = data['y']
    #plot(X,y)
    X = np.insert(X,0,values=np.ones(X.shape[0]),axis=1)
    theta = np.ones([1,2],dtype=int)
    cost = computeCost(theta,X,y,1.0)
    grad = gradient(theta,X,y,1.0)
    learningRate = 0.0
    fmin = minimize(fun=computeCost,x0=theta,args=(X,y,learningRate), method='TNC', jac=gradient)
    theta = np.array(fmin.x)
    print computeCost(theta,X,y,0.0)
    theta = np.reshape(theta,(1,2))
    #plotLine(X.transpose(),theta)
    tcost,vcost = learningCurve(data)
    plotCurve(tcost,vcost)
    pass

