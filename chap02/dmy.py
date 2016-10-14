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

X = pd.concat([df.const,df.sepalL,df.sepalW],axis=1);
#X = pd.concat([const,df.sepalL],axis=1)
print X.tail()
y = np.where(df.kind == 'setosa', 1, 0)
y1 = pd.Series(y);
Y = pd.DataFrame({'flag':y1});
print Y.tail()

model = sm.Logit(Y, X)
result = model.fit()

print result.summary()
