"""
Created on Tue Jan  9 21:21:54 2024

@author: chris
"""
# Implement and analyze a simple feedforward neural network using the ReLU
# activation function and backpropagation.

# Tasks:
# 1. Implement the ReLU activation function.
# 2. Complete the forward pass to compute the network output.
# 3. Compute the loss for the given input and target.
# 4. Implement backpropagation to compute the gradients with respect to:
#       - W1
#       - W2
#       - b1
#       - b2
# 5. Print the computed gradients.

import numpy as np

#Def relu
def relu(x):
    return np.maximum(0,x)

class NeuralNet:
	def __init__(self, W, B): 
		self.W = W
		self.B = B
		self.L = len(B)

	def evaln(self, x, N): 
		for n in range(0, N):
			x = relu(self.W[n]@x + self.B[n])
		return x

	def full_eval(self, x): 
		return self.evaln(x, self.L)

	def backprop(self, x, dcdx, N, bias=False): 
		z = self.W[-1]@self.evaln(x, self.L-1) + self.B[-1]
		deln = (relu(z) > 0)*dcdx
		for n in range(self.L-1, N-1, -1):
			z = self.W[n-1]@self.evaln(x, n-2) + self.B[n-1]
			deln = (relu(z) > 0)*(self.W[n].T@deln)
		return deln if bias else deln@self.evaln(x, N-1).T 

nn = NeuralNet((np.array([[-1, 1],[1, -2]]), np.array([[3, -1]])), (np.array([[1],[-1]]), np.array([[2]])))
x = np.array([[1],[2]])
y = np.array([[7]])
dfdx = nn.full_eval(x) - y 
print("Gradient of the cost function: ")
print("W1: \n" , nn.backprop(x, dfdx, 1))
print("W2: \n", nn.backprop(x, dfdx, 2))
print("b1: \n", nn.backprop(x, dfdx, 1, bias=True))
print("b2: \n", nn.backprop(x, dfdx, 2, bias=True))
