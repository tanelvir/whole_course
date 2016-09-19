import numpy as np
import matplotlib.pyplot as plt
# x and y should be vectors of equal length
# should return their correlation as a number

#I took this model from the course book page 61
def de_mean(x):
	x_bar = np.mean(x)
	return [x_i - x_bar for x_i in x]

#I took this model from the course book page
def corr(x,y):
	sd_x = np.std(x)
	sd_y = np.std(y)
	n = len(x)
	cov = np.dot(de_mean(x), de_mean(y))/(n-1)
	if sd_x > 0 and sd_y > 0:
		return cov/sd_x/sd_y
	else:
		return 0

#data = np.loadtxt('smoking.txt')
#print corr(data[:,0], data[:,1])