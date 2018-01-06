import numpy as np
import h5py
import matplotlib.pyplot as plt
from L_model_forward import L_model_forward
from L_model_backward import L_model_backward
from updateParameters import updateParameters
from computeCost import computeCost
from initializeParameters import initializeParameters
from Activations import relu,sigmoid

filename = 'Iris_mod.csv'
raw_data = open('Iris_mod.csv','rt')
data = np.loadtxt(raw_data, delimiter = ",")

np.random.seed(0)

idx_test = np.random.randint(150,size=5) #return 1-D array with 5 elements between 0 to 149(both inclusive). This will make test set
idx_train = np.arange(150)#create an array with all integers from 0 to 149
idx_train = np.delete(idx_train,idx_test)#remove indexes that go in test to get an array of indexes for train


test = data[idx_test,:] #each of 5 random rows is picked to make test set y
train = data[idx_train,:]

X_train = train[:,:4].T#each column is a new example; [:,:4] because 4 is not included, i.e. slice upto 4
Y_train = train[:,4]
Y_train = np.reshape(Y_train,(1,Y_train.shape[0]))

X_test = test[:,:4].T
Y_test = test[:,4]
Y_test = np.reshape(Y_test,(1,Y_test.shape[0]))

layer_dims = np.array([4,4,6,3])
learning_rate = 0.00001

for i in range(5000):
	parameters = initializeParameters(layer_dims)
	AL, caches = L_model_forward(X_train,parameters)
	grads = L_model_backward(AL,Y_train,caches)
	parameters = updateParameters(parameters, grads, learning_rate)	
print(computeCost(AL,Y_train))

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

print(computeCost(h,Y_test))

h = np.argmax(h,axis=0)+1
print(h)	

