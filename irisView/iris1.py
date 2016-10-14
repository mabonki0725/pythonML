import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#df = pd.read_csv('https://archive.ics.uci.edu/ml/'  \
# 'machine-learning-databases/iris/iris.data', header=None)
df=pd.read_csv('iris.csv',header=3,
 names=('no','sepalL','sepalW','petalL','petalW','kind'))
print df.kind.head()

y = df.iloc[0:100, 5].values

y = np.where(y == 'setosa', -1, 1)
X = df.iloc[0:150, [1,2,3]].values
#print X[0:50,]

#2D figure
plt.scatter(X[:50, 0], X[:50, 1], 
  color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
  color='blue', marker='x', label='versicolor')
plt.scatter(X[100:150, 0], X[100:150, 1],
  color='green', marker='*', label='versica')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

#3D figure
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(df.sepalL[:50],df.sepalW[:50],df.petalL[:50],
  color='red',marker='o',label='setosa')
ax.scatter3D(df.sepalL[50:100],df.sepalW[50:100],df.petalL[50:100],
   color='blue',marker='x',label='versicolor')
ax.scatter3D(df.sepalL[100:150],df.sepalW[100:150],df.petalL[100:150],
   color='green',marker='*',label='versica') 
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
ax.set_zlabel('petal length')
plt.show()




