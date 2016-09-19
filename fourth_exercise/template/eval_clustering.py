import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mark

#Just run the whole file
def kmeans(data, seedIndices): #run k-means
	cluster_center = [] #create empty matrix for  clusters
	for i in range(len(seedIndices)): #go through initial seed indices
		cluster_center.append(data[seedIndices[i]]) #store coordinates of seed indices
	cluster_vector = np.ones(len(data)) #initialize labels for each value
	times = 0 #how many rounds
	while (times < 50): #start training loop
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

	return cluster_center, cluster_vector

# input: 1) Datamatrix as loaded by numpy.loadtxt()
#        2) Random datamatrix of same size as input 1
#        3) numpy array of length 10 with initial center indices
# output: vector (numpy array) conisting of the gap statistics for k=1..10
# note this function should be called 10 times and averaged for the report.

def eval_clustering(data, randomData, initialCenters): #calculate gap vector (vector of gap statistics values)
	E_k_d = np.ones(len(initialCenters), dtype=float) #intial centers for given data (k-means objective function value)
	E_k_r = np.ones(len(initialCenters), dtype=float) #initial centers for random array (k-means objective function value)
	for k in range(len(initialCenters)): #go through every k
		cluster_center, cluster_vector = kmeans(data, initialCenters[:k+1]) #calculate k-means for this k (data array)
		E_k_d[k] = objective_function_kmeans(data, cluster_center, cluster_vector) #store k-means objective function value for array
		cluster_center, cluster_vector = kmeans(randomData, initialCenters[:k+1]) #calculate k-means for this k (data array)
		E_k_r[k] = objective_function_kmeans(randomData, cluster_center, cluster_vector) #store k-means objective function value for array
	gap_vector = gap_statistics(E_k_r, E_k_d) #call gap statistics
	return gap_vector

# input:  	1) datamatrix as loaded by numpy.loadtxt()
#			2) numpy array of the cluster centers.
#			3) vector (numpy array) consisting of the assigned cluster for each observation.
# output:	the k-means objective function value as specified in the assignment for the given k
# note k is infered based on number of elements in centers
def objective_function_kmeans(data,centers,clusterLabels): #call objective function
	tot_vec = 0.0 #initialize k-means objective function value
	for k in range(len(centers)): #go through centers
		for i in range(len(clusterLabels)): #go through label for each element
			if (k == clusterLabels[i]): #if it is a match
				tot_vec += np.linalg.norm(data[i] - centers[k])**2 #calculate euclidean distance
	return tot_vec

# input:  	1) vector (numpy array) of objective function values for k=1..10 for a random dataset
#			2) vector (numpy array) of objective function values for k=1..10 for a given dataset
# output:	vector (numpy array) of the computed gap statistics for each k=1..10
# note should calculate step 4 in 4.a in assignment description
def gap_statistics(Erand,E): #gap statistic, check the assignment 4 exc. 4
	return np.log(Erand)-np.log(E)

def create_random_array(arr): #creates random array
	lent = np.shape(arr)[1] #length of columns
	empty_arr = np.ones([np.shape(arr)[0], lent]) #create empty array
	for i in range(lent): #go through each column
		column = arr[:,i] #choose one column
		max = np.amax(column) #find max value
		min = np.amin(column) #find min value
		new_column = np.random.uniform(min, max, len(column)) #shuffle values in each column between min-max
		for j in range(len(empty_arr)): #set new values
			empty_arr[j,i] = new_column[j] #set new elements
	return empty_arr

np.random.seed(0)
data = np.loadtxt('Irisdata.txt') #set your own path for Iris data
rand_data = create_random_array(data) #creates random array
k = [1,2,3,4,5,6,7,8,9,10] #k [1..10]
gaps = np.zeros(10) #initialize gap statistics
for g in gaps: # go through 10 times
	gap = eval_clustering(data, rand_data, k) #call eval_clustering for each gap
	gaps += gap
gaps = [x / 10 for x in gaps] #take the average
print gaps
# colors = ["r", "b", "g", "c", "m", "y", "k", "w", "b", "r"]
# for i in range(len(gaps)):
# 	plt.scatter(gaps[i], gaps[i], c=colors[i], marker='o')
# plt.axis('equal')
# plt.show()