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

filename = 'Iris_mod.csv'
raw_data = open('Iris_mod.csv','rt')
data = np.loadtxt(raw_data, delimiter = ",")

np.random.seed(0)

idx_test = np.random.randint(150,size=30) #return 1-D array with 30 elements between 0 to 149(both inclusive). This will make test set
idx_train = np.arange(150)#create an array with all integers from 0 to 149
idx_train = np.delete(idx_train,idx_test)#remove indexes that go in test to get an array of indexes for train


test = data[idx_test,:] #each of 30 random rows is picked to make test set y
train = data[idx_train,:]

X_train = train[:,:4].T#each column is a new example; [:,:4] because 4 is not included, i.e. slice upto 4
Y_train = train[:,4]
Y_train = np.reshape(Y_train,(1,Y_train.shape[0]))
Y_train_b = np.zeros((3,Y_train.shape[1])) #binary matrix of Y_train
for l in range(0,Y_train.shape[1]):
	Y_train_b[Y_train[0,l].astype(int)-1,l] = 1

X_test = test[:,:4].T
Y_test = test[:,4]
Y_test = np.reshape(Y_test,(1,Y_test.shape[0]))
Y_test_b = np.zeros((3,Y_test.shape[1])) #binary matrix of Y_test
for l in range(0,Y_test.shape[1]):
	Y_test_b[Y_test[0,l].astype(int)-1,l] = 1

costs = []

layer_dims = np.array([4,5,6,3])
learning_rate = 0.15

parameters = initializeParameters(layer_dims)

for i in range(15000):
	AL, caches = L_model_forward(X_train,parameters)
	grads = L_model_backward(AL,Y_train_b,caches)
	parameters = updateParameters(parameters, grads, learning_rate)	
	costs.append(computeCost(AL,Y_train_b))

print(computeCost(AL,Y_train_b))
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

print(computeCost(h,Y_test_b))

h = np.argmax(h,axis=0)+1
print(h)	

