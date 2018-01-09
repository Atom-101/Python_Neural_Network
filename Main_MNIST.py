import numpy as np
#import h5py
import matplotlib.pyplot as plt
from L_model_forward import L_model_forward
from L_model_backward import L_model_backward
from updateParameters import updateParameters
#from computeCostMSE import computeCost
from computeCost import computeCost
from initializeParameters import initializeParameters
from Activations import relu,sigmoid
#from gradientCheck import gradientCheck


raw_data_train = open('train_mini.csv','rt')
data_train = np.loadtxt(raw_data_train, delimiter = ",")

raw_data_test = open('test_mini.csv','rt')
data_test = np.loadtxt(raw_data_test, delimiter = ",")


X_train = data_train[:,1:].T	#each column is a new example
Y_train = data_train[:,0]
Y_train = np.reshape(Y_train,(1,Y_train.shape[0]))
Y_train_b = np.zeros((10,Y_train.shape[1])) #binary matrix of Y_train
for l in range(0,Y_train.shape[1]):
	Y_train_b[Y_train[0,l].astype(int)-1,l] = 1

mu = (1/X_train.shape[1])*np.sum(X_train,axis =1)
sigma = (1/X_train.shape[1])*np.sum(X_train**2,axis =1)

#X_train = np.divide((X_train-mu),sigma)

X_test = data_test[:,:].T
#X_test = np.divide((X_test-mu),sigma)

"""
Y_test = data_test[:,0]
Y_test = np.reshape(Y_test,(1,Y_test.shape[0]))
Y_test_b = np.zeros((10,Y_test.shape[1])) #binary matrix of Y_test
for l in range(0,Y_test.shape[1]):
	Y_test_b[Y_test[0,l].astype(int)-1,l] = 1
"""
costs = []

layer_dims = np.array([784,60,10])
learning_rate = 0.0003
lambd = 0.1

parameters = initializeParameters(layer_dims)

for i in range(10):
	AL, caches = L_model_forward(X_train,parameters)
	grads = L_model_backward(AL,Y_train_b,caches,parameters,lambd)
	parameters = updateParameters(parameters, grads, learning_rate)	
	costs.append(computeCost(AL,Y_train_b,parameters,lambd))

print(computeCost(AL,Y_train_b,parameters,lambd))
plt.plot(costs)
plt.ylabel("cost")
plt.xlabel("number of iterations")
plt.show()	

#print(gradientCheck(parameters, grads, .001, X_test, Y_test))

A = X_test
for l in range(1,len(parameters)//2):
	#print(A.shape)
	#print(parameters['W'+str(l)].shape)
	#print(parameters['b'+str(l)])
	Z = np.dot(parameters['W'+str(l)],A)+parameters['b'+str(l)]
	A,Z = relu(Z)
	#print(l)

Z = np.dot(parameters['W'+str(l+1)],A)+parameters['b'+str(l+1)]
h,Z = sigmoid(Z)

#print(computeCost(h,Y_test_b))

h = np.argmax(h,axis=0)
#print(h)	

