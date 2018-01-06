import numpy as np
import h5py
import matplotlib.pyplot as plt

def linearForward(A,W,b):
	Z = np.dot(W,A)+b
	cache = (A,W,b)
	return Z,cache