import numpy as np
import h5py
import matplotlib.pyplot as plt
from linearActivationForward import linearActivationForward

def L_model_forward(X, parameters):
	caches = []
	A = X
	L = len(parameters)//2
	for l in range(1,L):
		A_prev = A
		A, cache = linearActivationForward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],"relu")
		caches.append(cache)
		#print(l)
	
	AL, cache = linearActivationForward(A,parameters['W'+str(L)],parameters['b'+str(L)],"sigmoid")
	caches.append(cache)
	
	return AL, caches