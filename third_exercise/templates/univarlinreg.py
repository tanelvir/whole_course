import math
import numpy as np
# input: 1) x: the independent variable, as a N dimensional vector as a numpy array
#        2) y: the dependent variable, as a N dimensional vector as a numpy array
#
# output: 1) the alpha parameter
#         2) the beta parameter


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

age = np.loadtxt('smoking.txt')
fev = np.loadtxt('smoking.txt')
age = age[:,0]
fev = fev[:,1]

print univarlinreg(age, fev)