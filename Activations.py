import numpy as np 


def sigmoid(Z):
	A = 1/(1+np.exp(-Z))	
	A = np.reshape(A,(Z.shape))
	return A,Z

def relu(Z):
	A = np.maximum(Z,0)
	return A,Z

def sigmoidBackward(dA, activation_cache):
	temp = np.exp(-activation_cache)
	temp = np.nan_to_num(np.divide(-1*temp,(1+temp)**2))
	dZ = np.multiply(dA,temp)
	return dZ
	
def reluBackward(dA, activation_cache):
	temp = 1*(activation_cache>0)
	dZ = np.multiply(dA,temp)
	return dZ
	
	