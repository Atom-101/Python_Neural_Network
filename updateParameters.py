import numpy as np
import h5py
import matplotlib.pyplot as plt

def updateParameters(parameters, grads, learning_rate):
	
	L =len(parameters)//2
	
	for l in range(1,L):
		parameters["W"+str(l)] = parameters["W"+str(l)]-learning_rate*grads["dW"+str(l)]
		parameters["b"+str(l)] = parameters["b"+str(l)]-learning_rate*grads["db"+str(l)]
		#print(parameters["b"+str(l)].shape)
	return parameters