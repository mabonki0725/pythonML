import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

#regression
df=pd.read_csv('iris1.csv',header=3,
 names=('no','const','sepalL','sepalW','petalL','petalW','kind'))
#print df.kind.head()
#print df.describe()
print df.groupby('kind')['no'].size()

#wrk = pd.Series(np.ones(df.shape[0]))
#const = pd.DataFrame({'const':wrk});
#X = pd.concat([const,df.sepalL,df.sepalW,df.petalL,df.petalW],axis=1)
X = pd.concat([df.const,df.sepalW,df.petalL],axis=1)
#X = pd.DataFrame({'const':wrk},df.sepalL,df.sepalW);
#X = pd.concat([const,df.sepalL],axis=1)
print X.tail()
y = np.where(df.kind == 'setosa', 1, 0)
y1 = pd.Series(y);
Y = pd.DataFrame({'flag':y1});
print Y.tail()

plt.scatter(df[y1 == 1]['sepalW'],df[y1 == 1]['petalL'],
 color='red', marker='*', label='setosa')
plt.scatter(df[y1 == 0]['sepalW'],df[y1 == 0]['petalL'],
 color='blue', marker='o', label='else') 
plt.legend(loc='upper left')

plt.show()

model = sm.Logit(Y, X)
result = model.fit()



print result.summary()

