import numpy as np
#import h5py
import matplotlib.pyplot as plt

def initializeParameters(layer_dims):
	parameters = {}
	L = len(layer_dims)
	for l in range(1,L):
		np.random.seed(1)
		parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
		np.random.seed(3)#each random seed affects only the immediately next random call
		parameters['b'+str(l)] = np.random.randn(layer_dims[l],1)*np.sqrt(2/layer_dims[l-1])
	return parameters
