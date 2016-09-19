import numpy as np
import matplotlib.pyplot as plt
# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return a tuple containing average FEV1 of smokers and nonsmokers 
def meanFEV1(data):
	non_smoker_list = []
	smoker_list = []
	for i in range(len(data)):
		if data[i,4] == 0.0:
			non_smoker_list.append(data[i,1])
		else:
			smoker_list.append(data[i,1])
	non_mean = np.mean(non_smoker_list)
	smoker_mean = np.mean(smoker_list)
	#print smoker_mean, non_mean
	return smoker_mean, non_mean

#meanFEV1(np.loadtxt('smoking.txt'))