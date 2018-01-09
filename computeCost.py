import numpy as np
#import h5py
import matplotlib.pyplot as plt

def computeCost(AL, Y, parameters,lambd):
	W = np.array([])
	for l in range(1,len(parameters)//2+1):
		temp = parameters["W"+str(l)]
		W= np.append(W,temp)
	#W = np.array(W)
	#W = W.reshape(1,W.shape[0])
	#print(W.shape)
	#print(W)
	m = Y.shape[1]
	cost = (-1/m)*np.sum(np.dot(np.log(AL),Y.T)+np.dot(np.log(1-AL),(1-Y).T))+(lambd/(2*m))*np.sum(W**2)

	return cost
