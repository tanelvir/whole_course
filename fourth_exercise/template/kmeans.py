import numpy as np
# input:  	1) datamatrix as loaded by numpy.loadtxt('Irisdata.txt').
#			2) vector (numpy array) of indices for k seed observation.
# output: 	1) numpy array of the resulting cluster centers. 
#			2) vector (numpy array) consisting of the assigned cluster for each observation.
# note the indices are an index to an observation in the datamatrix

def kmeans(data, seedIndices): #run k-means
	cluster_center = [] #create empty matrix for  clusters
	for i in range(len(seedIndices)): #go through initial seed indices
		cluster_center.append(data[seedIndices[i]]) #store coordinates of seed indices
	cluster_vector = np.ones(len(data)) #initialize labels for each value
	times = 0 #how many rounds
	while (times < 500): #start training loop
		times += 1 #increase rounds
		for i in range(len(data)): #assign every value into a cluster
			temp_best = 100000 #initialize distance
			best_k = 0 #initialize best k
			for k in range(len(cluster_center)): #go through k-clusters
				temp = np.linalg.norm(data[i] - cluster_center[k]) #Euclidean distance
				if temp <= temp_best: #if we have a new closer neighbor
					temp_best = temp #assign a new distance
					best_k = k #assign a new k
			cluster_vector[i] = best_k #assign value to the closest cluster

		for c in range(len(cluster_center)): #assign a new cluster centers
			temp_vec = [] #initialize vector for mean calculation
			for d in range(len(cluster_vector)): #go through every value
				if (cluster_vector[d] == c): #if this element belongs in a cluster
					temp_vec.append(data[d]) #add to the mean calculation
			cluster_center[c] = np.mean(temp_vec, axis=0) #calculate the mean

	print cluster_center #see the new centers
	print cluster_vector #see the labels for each element
	return cluster_center, cluster_vector

np.random.seed(0)
data = np.loadtxt('Irisdata.txt') #set your own path for Iris data
k = [1,5,9] #set some start indices (cluster center coordinates)
kmeans(data, k) #run the code