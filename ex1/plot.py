import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab
df = pd.read_csv('ex1data1.txt',header=-1)
x = df[0]
y = df[1]
plt.scatter(x,y,marker='x',color='r')
plt.title('Population VS Profit')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of city in 10,000s')
plt.xlim(4,24)
plt.ylim(-5,25)
plt.show()
