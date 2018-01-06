import numpy as np
import h5py
import matplotlib.pyplot as plt

def linearBackward(dZ,cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]
	
	dW = (1/m)*np.dot(dZ,A_prev.T)
	db = (1/m)*np.sum(dZ,axis =1,keepdims= True )#sum along columns to get column vector
	#print(db)
	#print(db.shape)
	dA_prev = np.dot(W.T,dZ)
	
	return dA_prev, dW, db