import numpy as np
import math
import matplotlib.pyplot as plt

def train(X1,xout1,T1,l):
	phi = []
	phix = []

	for i in range(7):
		phi1 = 1
		phi2 = math.exp(-10*(X1[i]-0.5)**2)
		phi3 = math.exp(-10*(X1[i]+0.5)**2)
		phi.append([phi1,phi2,phi3])

	for i in range(7):
		phi1 = 1
		phi2 = math.exp(-10*(xout[i]-0.5)**2)
		phi3 = math.exp(-10*(xout[i]+0.5)**2)
		phix.append([phi1,phi2,phi3])


	phi = np.array(phi)
	phix = np.array(phix)

	phixT = phix.T
	phiT = phi.T

	print phi
	print phiT

	phiphiT = phiT.dot(phi)
	phixphixT = phixT.dot(phix)

	print phiphiT

	lbda = l

	reg = np.identity(3)
	reg = lbda*reg

	print reg

	u = phiphiT + reg

	print u

	v = np.linalg.inv(u)

	b = phi.dot(v)

	b = b.T

	w = b.dot(T1)

	y = phi.dot(w)
	
	return y,y




X = np.linspace(-1.0,1,21)

xout = np.linspace(-0.95,0.95,21)

T = np.array([5.12,4.97,4.92,4.83,4.90,5.06,5.29,5.34,5.36,5.76,5.99,6.30,6.66,6.70,7.49,7.92,8.48,9.09,9.7,10.3,10.98])

truearray = np.array([5,4.92,4.88,4.88,4.92,5,5.12,5.28,5.48,5.72,6,6.32,6.68,7.08,7.52,8,8.52,9.08,9.68,10.32,11])

tmean = np.mean(truearray)


print 'X is ', X
print 'T is ', T

X1 = []
xout1 = []
X2 = []
xout2 = []
X3 = []
xout3 = []
T1 = []
T2 = []
T3 = []

i = 0

while i<21:
	X1.append(X[i])
	xout1.append(xout[i])
	T1.append(T[i])
	X2.append(X[i+1])
	xout2.append(xout[i+1])
	T2.append(T[i+1])
	X3.append(X[i+2])
	xout3.append(xout[i+2])
	T3.append(T[i+2])
	i += 3

X1 = np.array(X1)
T1 = np.array(T1)
X2 = np.array(X2)
T2 = np.array(T2)
X3 = np.array(X3)
T3 = np.array(T2)
xout1 = np.array(xout1)
xout2 = np.array(xout2)
xout3 = np.array(xout3)

print 'X1 is ', X1
print 'T1 is ', T1

print 'X2 is ', X2
print 'T2 is ', T2

print 'X3 is ', X3
print 'T3 is ', T3

yr = []
xd = []
xr = []

xr = np.array([])
xd = np.array([])
yr = np.array([])

X1 = np.append(X1,X2)
X1 = np.append(X1,X3)
xd = np.append(xd,X1)

for i in range(10):
	Y1,yr1 = train(X1,xout1,T1,i)
	Y2,yr2 = train(X2,xout2,T2,i)
	Y3,yr3 = train(X3,xout3,T3,i)
	Y1 = np.append(Y1,Y2)
	Y1 = np.append(Y1,Y3)
	yr = np.append(yr,Y1)
	xr = np.append(xr,xd)
	print np.shape(xr)
	print np.shape(yr)

print 'final',np.shape(xr)
print 'final',np.shape(yr)


sc = plt.scatter(X,T)
plt.plot(X1,Y1)
	
plt.show()