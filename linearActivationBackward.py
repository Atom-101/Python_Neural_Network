import numpy as np
import h5py
import matplotlib.pyplot as plt
from Activations import reluBackward, sigmoidBackward
from linearBackward import linearBackward

def linearActivationBackward(dA,cache,activation):
	linear_cache, activation_cache = cache
	if (activation == "sigmoid"):
		dZ = sigmoidBackward(dA, activation_cache)
	elif (activation == "relu"):
		dZ = reluBackward(dA, activation_cache)
	dA_prev,dW,db = linearBackward(dZ, linear_cache)
	
	return dA_prev,dW,db