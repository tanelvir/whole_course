import numpy as np
# input: 1) x: the independent variables (data matrix), as a N x M dimensional matrix as a numpy array
#        2) y: the dependent variable, as a N dimensional vector as a numpy array
#
# output: 1) the regression coefficients as a (M+1) dimensional vector as a numpy array
#
# note: the regression coefficients should include the w_0 (the free parameter), thus having dimension (M+1).
# note: The tested datamatrix is **NOT** extended with a column of 1's - if you prefer to do this, then you can do it inside the function by extending x.       
def multivarlinreg(x, y):
	y_points = len(y)
	onevec = np.ones((y_points,1))
	X = np.concatenate((onevec, x), axis = 1)
	w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
	return w

data = np.loadtxt('smoking.txt')
fev = data[:,1]
data1 = data[:,0]
data_rest = data[:,2:]
onevec1 = np.ones((len(data1),1))
for t in range(len(data1)):
	onevec1[t] = data1[t]
final = np.concatenate((onevec1, data_rest), axis = 1)
print multivarlinreg(final, fev)