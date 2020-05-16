#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('My.csv')

X = data.drop(columns = ['No', 'X1 transaction date'])
#y = data['house price of unit area']

x = np.array(X)
#y = np.array(y)
#x=np.array([[0,0,1,1],[1,0,0,0],[0,1,1,0],[0,0,0,1]])
w=np.random.randn(x.shape[1], 2)

scaler = StandardScaler()
x = scaler.fit_transform(x)

lr=0.5
epoch=5
cluster=[]
for iter in range(epoch):
    for i in range(x.shape[0]):
        temp=[]
        #print("for input",x[i,:])
        for j in range(w.shape[1]):
            sum=0
            for k in range(w.shape[0]):
                sum=sum+((x[i][k]-w[k][j])**2)
            temp.append(sum)
        if(temp[0]<temp[1]):
            index=0
            cluster.append(index)
        else:
            index=1
            cluster.append(index)
        #print("neuron at position",index,"won")
        for p in range(w.shape[0]):
            w[p][index]=w[p][index]+lr*(x[i][p]-w[p][index])
        #print(w)
    lr=lr*0.5
print(cluster)


# In[2]:


np.shape(cluster)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




