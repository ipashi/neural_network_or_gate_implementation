import numpy as np 
from random import random
from random import seed
seed(1)
data = [[1,1,1],
         [1,0,1],
         [0,1,1],
         [0,0,0]]
weights = [random() for i in range(3)]
chage_weights=[]

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def sigmoid_derviate(x):
	return x*(1.0-x)

def activation_input(data_i,weights_i):
	act=weights[-1]
	for i in range(len(data_i)-1):
		act+=data_i[i]*weights_i[i]
	return act
learning_rate = 0.7

def forward(data,weights,predict):
	chage_weights = weights
	for _ in range(50000):
		error=0.0
		count = 0
		for i in data:
			output = sigmoid(activation_input(i,chage_weights))
			error = i[-1] - output
			error_square = error**2
			delta = error*sigmoid_derviate(output)
			for j in range(len(i)-1):
				chage_weights[j] = chage_weights[j] + delta*i[j]*learning_rate
			chage_weights[-1]= chage_weights[-1] + delta*learning_rate
			count = count+1
			# print(output,count,sep=" : ")
	out = sigmoid(activation_input(predict,chage_weights))
	print("predicted value is : {},{}".format(int(round(out))))
predict = [1,0,"check"]
forward(data,weights,predict)



