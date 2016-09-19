import numpy as np
import heapq as hp
# input: train: 1) train data (without labels) in the form of a N by d numpy array, where N is the number of training data points and d is the number of dimensions
#               2) test: test data (without labels) in the form of a M by d numpy array, where M is the number of test data points and d is the number of dimensions
#               3) trainlabels: labels for training data in the form of a N by 1 numpy vector, where N is the number of training data points
#               4) k: paramater k
# output:1) distance matrix (numpy array) between test and training samples 
#        2) vector (numpy array) consisting of the predicted classes for the test data
#
# note: the labels should **not** be part of the train/test data matrices!

def knn(train, test, trainlabels, k):
    dis_matrix = []
    classes = []
    min = k
    for i in range(len(test)):
        whole_line = []
        for j in range(len(train)):
            whole_line.append(np.linalg.norm(test[i]-train[j]))
        dis_matrix.append(whole_line)

    for i in range(len(dis_matrix)):
        results = hp.nsmallest(k, range(len(dis_matrix[i])), dis_matrix[i].__getitem__)
        total = 0
        print results
        for r in results:
            total += trainlabels[r]
        #print total
        if total > k/2:
            classes.append(1)
        else:
            classes.append(0)

    #print dis_matrix
    #print classes
    return dis_matrix, classes

# data_train = np.loadtxt('parkinsonsTrain.dt')
# data_test = np.loadtxt('parkinsonsTest.dt')
# #print data_train[0]
# #print data_test[0]
# labels = np.full(98, 1)
# k = 4
# knn(data_train, data_test, labels, k)