import numpy as np
#import h5py
import matplotlib.pyplot as plt

def computeCost(AL, Y):
	m = Y.shape[1]
	cost = (-1/m)*np.sum(np.dot(np.log(AL),Y.T)+np.dot(np.log(1-AL),(1-Y).T))

	return cost
