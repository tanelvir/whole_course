import math
import numpy as np
import random
# input: 1) x: the independent variable, as a N dimensional vector as a numpy array
#        2) y: the dependent variable, as a N dimensional vector as a numpy array
#        3) alpha: the alpha parameter
#        4) beta: the beta parameter
#
# output: 1) the root mean square error (rmse) 

def new_mean(arr):
	sum = 0.0
	for i in arr:
		sum += i
	mean = float(sum/len(arr))
	return mean

def dot(x,y):
	return sum(p*q for p,q in zip(x, y))

def univarlinreg(x,y):
	N = len(x)
	beta_top = new_mean(x)*new_mean(y) - dot(x,y)/N
	beta_bottom = new_mean(x)**2 - dot(x,x)/N
	beta = beta_top/beta_bottom
	alpha = new_mean(y)-beta*new_mean(x)
	return alpha, beta

def multivarlinreg(x, y):
	y_points = len(y)
	onevec = np.ones((y_points,1))
	X = np.concatenate((onevec, x), axis = 1)
	w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
	return w

def rmse(x, y, alpha, beta):
	sum = 0
	for i in range(len(x)):
		sum += abs(y[i]-(alpha+beta*x[i]))**2
	return math.sqrt(float(sum/len(x)))

data = np.loadtxt('smoking.txt')
new_rand = random.shuffle(data)
test = data[:450,:]
train = data[450:,:]
fev_train = train[:,1]
age_train = train[:,0]
fev_test = test[:,1]
age_test = test[:,0]

a_train, b_train = univarlinreg(age_train, fev_train)
vector_train = multivarlinreg(train, fev_train)

a_test, b_test = univarlinreg(age_test, fev_test)
vector_test = multivarlinreg(test, fev_test)


print rmse(vector_train, vector_test, a_train, b_train)
# age1 = data[:450,0]
# fev1 = data[:450,1]
# age2 = data[450:,0]
# fev2 = data[450:,1]
#
# print len(age1)
# print len(age2)
#
# a,b = univarlinreg(age1, fev1)
#
# print rmse(age2, fev2, a, b)