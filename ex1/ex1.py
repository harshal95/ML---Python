import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(df):
    # Plot the feature vs output values
    x = df[0]
    y = df[1]
    plt.scatter(x, y, marker='x', color='r')
    plt.title('Population VS Profit')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of city in 10,000s')
    plt.xlim(4, 24)
    plt.ylim(-5, 25)
    plt.draw()

def plot_line(x,theta):
    plt.scatter(x[1],np.dot(theta,x),marker='*',color='b')
    plt.show()


def costFun(x,y,theta):

# Find h(x) = a0*x0 + a1*x1 where a = theta_values, x = feature_values
# Next calculate the sum of the difference of squares of h(x) and y and divide by 2m

    h_x = np.dot(theta, x)
    diff = np.subtract(h_x, y)
    diff_sq = np.square(diff)
    cost_sum = np.sum(diff_sq)
    cost_fn = cost_sum / (2 * df_len)
    return cost_fn

def gradientDescent(x,y,theta,alpha,iterations):
    m = y.size
    j_history = np.zeros((iterations,1))
    for i  in range(iterations):
        h_x = np.dot(theta, x)
        diff = np.subtract(h_x, y)
        fst = np.multiply(diff,x[0])
        sec = np.multiply(diff,x[1])
        fst_sum = np.sum(fst)
        sec_sum = np.sum(sec)
        fst_quant = (fst_sum*alpha)/m
        sec_quant = (sec_sum * alpha)/m
        theta[0][0] = theta[0][0] - fst_quant
        theta[0][1] = theta[0][1] - sec_quant
        j_history[i][0] = costFun(x,y,theta)

    return j_history,theta


# Read dataset and get the features and output values in dataframe
# Reshape and concatenate ones to feature array for x0 and transpose the arrays for convenience
# Call the cost function
if __name__ == '__main__':
    df = pd.read_csv('ex1data1.txt',header=-1)
    plot(df)
    df_len = len(df)
    x = np.reshape(df[0].values,(df_len,1))
    y = np.reshape(df[1].values,(97,1))
    x0 = np.ones((df_len,1))
    x = np.concatenate([x0,x],axis=1)
    x = x.transpose()
    y = y.transpose()
    theta = np.zeros((1,2))
    cost_val = costFun(x,y,theta)
    print "The cost function value is ",cost_val
    cost_val = costFun(x, y,np.array([-1,2]))
    print "The cost function value is ", cost_val
    iterations = 1500
    alpha = 0.01
    cost_arr,theta = gradientDescent(x,y,theta,alpha,iterations)
    print "Expected theta values are ",theta
    plot_line(x,theta)





