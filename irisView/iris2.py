import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

#regression
df=pd.read_csv('iris.csv',header=3,
 names=('no','sepalL','sepalW','petalL','petalW','kind'))
print df.kind.head()
#print df.describe()
print df.groupby('kind')['no'].size()

xr = df.sepalL
yr = df.petalL

const = pd.Series(np.ones(df.shape[0]))
Xr = pd.DataFrame({'const':const,'sepalL':xr})

print Xr.head()

model = sm.OLS(yr, Xr)
result = model.fit()
print result.summary()

#draw figure
xRange = np.sqrt((np.max(xr)-np.min(xr))**2)
xMin = np.min(xr)-0.05*xRange
xMax = np.max(xr)+0.05*xRange
xLine = np.linspace(xMin, xMax, 50)
xLine = pd.DataFrame(xLine)

beta = result.params
yLine = beta[0] + beta[1]*xLine

yRange = np.sqrt((np.max(yr)-np.min(yr))**2)
yMin = np.min(yr)-0.1*yRange
yMax = np.max(yr)+0.3*yRange

plt.scatter(Xr[df.kind=='setosa']['sepalL'], yr[df.kind =='setosa'], 
            facecolors='none', edgecolors='r', label='setosa')
plt.scatter(Xr[df.kind=='versicolor']['sepalL'], yr[df.kind=='versicolor'], 
            facecolors='none', edgecolors='b', label='versicolor')
plt.scatter(Xr[df.kind=='virginica']['sepalL'], yr[df.kind=='virginica'], 
            facecolors='none', edgecolors='g', label='virginica')
plt.plot(xLine, yLine, 'k-')
plt.legend(loc='upper left', scatterpoints=1)
plt.xlabel("sepalL")
plt.ylabel("sepalW")
plt.ylim(yMin, yMax)
plt.show()
