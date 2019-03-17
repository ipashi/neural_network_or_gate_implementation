import numpy as np 
from random import random
from random import seed

#seed to get same random for each time you run
seed(1)

#input data with last row value row[-1] as output value
data = [[1,1,1],
         [1,0,1],
         [0,1,1],
         [0,0,0]]

#initial random weights including bias
weights = [random() for i in range(3)]
chage_weights=[]

#activation fumction
def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

#derivative of sigmoid function
def sigmoid_derviate(x):
	return x*(1.0-x)

#function to calculate  sum(input*weight)+bias
def activation_input(data_i,weights_i):
	act=weights[-1]
	for i in range(len(data_i)-1):
		act+=data_i[i]*weights_i[i]
	return act
#learning rate aka as step size in stochastic gradient descent (back propagation)
learning_rate = 0.7

#moving forward in network and updating weights with back propagation and predicts output as well
def forward(data,weights,predict):
	chage_weights = weights
	for _ in range(50000):
		error=0.0
		for i in data:
			output = sigmoid(activation_input(i,chage_weights))
			error = i[-1] - output
			delta = error*sigmoid_derviate(output)
			for j in range(len(i)-1):
				chage_weights[j] = chage_weights[j] + delta*i[j]*learning_rate
			chage_weights[-1]= chage_weights[-1] + delta*learning_rate
	out = sigmoid(activation_input(predict,chage_weights))
	print("predicted value is : {},{}".format(int(round(out))))

predict = [1,0,"check"]
#call the function to predict the output
forward(data,weights,predict)



