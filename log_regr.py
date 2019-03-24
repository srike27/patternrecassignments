#!/usr/bin/env python
# coding: utf-8

# In[175]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
d1 = [[1.08,.08],[0.75,-0.19],[0.85,-0.11],[0.94,0.01],[0.4,-0.09],[1.25,-.21],[1.19,0.07],[0.99,0.04],[0.69,-0.02],[1.32,0.02]]
d2 = [[0.01,0.85],[-0.01,1.05],[0.09,0.93],[-0.05,1.41],[-0.45,1.45],[0.07,1.20],[-0.33,0.88],[-0.06,1.08],[-0.33,1.1],[-0.24,1.01]]
d1 = np.array(d1)
d2 = np.array(d2)
d1train = d1[0:5]
d2train = d2[0:5]
d1test = d1[5:10]
d2test = d2[5:10]
d = np.concatenate((d1,d2),0)
dtest = np.concatenate((d1test,d2test),0)
print(d)
print(d[1,:])
dtrain = np.concatenate((d1train,d2train),0)
print('Training data is :\n', dtrain)
dtrainx = dtrain[:,0]
dtrainy = dtrain[:,1]
w1 = [0,0]
w2 = [0,0]
w1 = np.array(w1)
w2 = np.array(w2)
T = [[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]
T = np.array(T, dtype = float)
plt.scatter(x = dtrainx, y = dtrainy)
print('scatter plot is:')
plt.show()


# In[176]:


def compute_prob(xk,w1,w2):
    e1 = xk.dot(w1)
    e2 = xk.dot(w2)
    s = np.exp(e1) + np.exp(e2)
    p = [np.exp(e1)/s,np.exp(e2)/s]
    p = np.array(p)
    return p


# In[177]:


def compute_y(d,w1,w2):
    y = np.zeros([int(d.size/2),2])
    for i in range(int(d.size/2)):
        x = d[i,:]
        y[i,:] = compute_prob(x,w1,w2)
    print(y)
    return y


# In[178]:


def compute_cross_entropy(y,t):
    J = 0
    for i in range(2):
        for j in range(int(y.size/2)):
            J -= t[j][i]*np.log(y[j][i])
    return J
            


# In[179]:


def g_descent(t,x,w1,w2,eta):
    w1n = w1
    w2n = w2
    
    for i in range(int(x.size/2)):
        y = compute_prob(x[i,:],w1n,w2n)
        w1n = w1n + eta*(float(t[i,0])-y[0])*x[i,:]
        w2n = w2n + eta*(float(t[i,1])-y[1])*x[i,:]
    return w1n,w2n


# In[ ]:





# In[180]:


w1x = w1
w2x = w2
J = np.zeros(20)
for i in range(20):
    w1x,w2x = g_descent(T,dtrain,w1x,w2x,0.1)
    y = compute_y(dtrain,w1x,w2x)
    J[i] = compute_cross_entropy(y,T)

print('Test data')
y1 = compute_y(dtest,w1x,w2x)

plt.plot(range(20),J)
plt.show


# In[ ]:





# In[ ]:





# In[ ]:




