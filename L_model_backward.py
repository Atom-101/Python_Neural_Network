import numpy as np
import h5py
import matplotlib.pyplot as plt
from linearActivationBackward import linearActivationBackward

def L_model_backward(AL,Y,caches):
	
	grads ={}
	#Y = Y.reshape(1,AL.shape[1])
	L = len(caches)
	
	dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))
	dAL = np.nan_to_num(dAL)
	
	current_cache = caches[L-1]
	grads["dA" + str(L)], grads["dW"+str(L)], grads["db"+str(L)] = linearActivationBackward(dAL, current_cache,"sigmoid")
	
	for l in range(L-2,-1,-1): #l runs from L-2 to 0(both inclusive)
		#caches vary from 0 to L-1, grads vary from 1 to L
		current_cache = caches[l]
		grads["dA" + str(l+1)], grads["dW"+str(l+1)], grads["db"+str(l+1)] = linearActivationBackward(grads["dA"+str(l+2)], current_cache,"relu")
	
	return grads