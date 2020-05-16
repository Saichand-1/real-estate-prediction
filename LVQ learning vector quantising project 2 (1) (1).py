#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('My.csv')
x=data.drop(columns = ['No', 'X1 transaction date'])
x=np.array(x)
w=np.random.randn(x.shape[1], 2)
t=data['Y house price of unit area']
t=np.array(t)
t=t.reshape(t.shape[0],1)
print(t.shape)
scaler = StandardScaler()
t = scaler.fit_transform(t)
#print(t)
lr=0.5
epoch=2
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
        #print("neuron at position",index,"won and target is",t[i])
        if index==t[i]:
            for p in range(w.shape[0]):
                w[p][index]=w[p][index]+lr*(x[i][p]-w[p][index])
        else:
            for p in range(w.shape[0]):
                w[p][index]=w[p][index]-lr*(x[i][p]-w[p][index])
        #print(w)
    lr=lr*0.5
print(cluster)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




