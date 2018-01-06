import numpy as np
import h5py
import matplotlib.pyplot as plt
from Activations import sigmoid,relu
from linearForward import linearForward

def linearActivationForward(A_prev,W,b,activation):
	Z, linear_cache = linearForward(A_prev, W, b)
	if (activation == "relu"):
		A, activation_cache = relu(Z)
	elif (activation == "sigmoid"):
		A, activation_cache = sigmoid(Z)
	cache = (linear_cache, activation_cache)
	return A, cache