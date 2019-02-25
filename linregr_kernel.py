import numpy as np
import math as m
import matplotlib.pyplot as plt

def kernelf(x1,x2,ss):
	y = m.exp((-(x1-x2)**2.0)/ss)
	return y



Xi = [-1,-0.7,-0.4,-0.1,0.2,0.5,0.8]
Vi = [-0.9,-0.6,-0.3,0,0.3,0.6,0.9]
testi = [-0.8,-0.5,-0.2,0.1,0.4,0.7,1]
traino = [5.12,4.83,5.29,5.76,6.66,6.70,8.48,10.30]
Vo = [4.97,4.9,5.34,5.99,6.7,8.48,10.30]
testo = [4.92,5.06,5.36,6.30,7.49,9.09,10.98]

Xi =np.array(Xi)
Vi =np.array(Vi)
testi =np.array(testi)
traino =np.array(traino)
Vo =np.array(Vo)
testo =np.array(testo)
ss = 0.1

Yi =[]
for j in range(Vi.size):
	M = []
	for i in range(Xi.size):
		mo = kernelf(Xi[i],Vi[j],ss)
		M.append(mo)
	print M
	mm = np.sum(M)
	print mm
	M = M/mm
	print M
	Y = M.dot(testo)
	Yi.append(Y)

Yi = np.array(Yi)

dif = Vo-Yi

SSE = dif.dot(dif)
print SSE

O = []
Yii = []

SS = np.array([0.2,0.3,0.4,0.5])
sse  = []
for k in range(SS.size):
	Yii = []
	for j in range(Vi.size):
		M = []
		for i in range(Xi.size):
			mo = kernelf(Xi[i],Vi[j],SS[k])
			M.append(mo)
		mm = np.sum(M)
		M = M/mm
		Y = M.dot(testo)
		Yii.append(Y)
	O = np.array(Yii)

	dif = Vo-O

	SSE = dif.dot(dif)

	sse.append(SSE)

print 'sse array'
print sse


Yi =[]
for j in range(Vi.size):
	M = []
	for i in range(Xi.size):
		mo = kernelf(Xi[i],Vi[j],SS[0])
		M.append(mo)
	print M
	mm = np.sum(M)
	M = M/mm
	Y = M.dot(testo)
	Yi.append(Y)

print 'training out'
print testo
print 'predicted out'
print Yi

Yi = np.array(Yi)

dif = Vo-Yi

SSE = dif.dot(dif)
print 'minimum sse'
print SSE

