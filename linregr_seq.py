import numpy as np
import math as m
import matplotlib.pyplot as plt

def train(X,T,W,eta):
	phi = []
	wnew = W
	for i in range(int(X.size*0.66)):
		phi = np.array([1,m.exp(-10.0*(X[i]-1.0)**2),m.exp(-10.0*(X[i]+1.0)**2)])
		e = (T[i] - phi[0]*wnew[0] - phi[1]*wnew[1] -phi[2]*wnew[2])
		wnew = wnew + eta*e*phi
	return wnew

def compSSE(X,W,T):
	i = (int(X.size*0.66))+1
	e =0	
	for i in range(X.size):
		phi = np.array([1,m.exp(-10.0*(X[i]-1.0)**2),m.exp(-10.0*(X[i]+1.0)**2)])
		e += (T[i] - phi[0]*W[0] - phi[1]*W[1] -phi[2]*W[2])**2
	return e

X = np.linspace(-1.0,1,21)

xout = np.linspace(-0.95,0.95,21)

T = np.array([5.12,4.97,4.92,4.83,4.90,5.06,5.29,5.34,5.36,5.76,5.99,6.30,6.66,6.70,7.49,7.92,8.48,9.09,9.7,10.3,10.98])

truearray = np.array([5,4.92,4.88,4.88,4.92,5,5.12,5.28,5.48,5.72,6,6.32,6.68,7.08,7.52,8,8.52,9.08,9.68,10.32,11])

tmean = np.mean(truearray)

W = np.zeros(3)

SSE = []
for i in range(10):
	W = train(X,T,W,0.4)
	SSE.append(compSSE(X,W,T))

print W 

Y = []
plt.plot(range(10),SSE)
"""for i in range(xout.size):
	phi = (np.array([1,m.exp(-10.0*(xout[i]-1.0)**2),m.exp(-10.0*(xout[i]+1.0)**2)]))
	Yi = phi.dot(W)
	Y.append(Yi)

plt.plot(xout,Y)
plt.plot(X,truearray)
"""
plt.show()
