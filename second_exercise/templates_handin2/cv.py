import numpy as np
from sklearn import cross_validation
import heapq as hp
from sklearn import preprocessing
# input:  1) training data in the form of a N by d numpy array, where N is the number of training data points and d is the number of dimensions
#         2) training labels in the form of a N by 1 numpy vector, where N is the number of training data points
#         3) a random permutation of entries as a numpy array, e.g. np.random.permutation(len(trainlabels))
# output: 1) the optimal k
#         2) an error matrix (numpy array) of size (5,13) where column i consists of the accuracy for the 5 folds for k=i
#
# The random-permuted vector rand_perm should be used for generating 5 folds, where the first fold consists of the first N/5 elements from rand_perm, rounded up to the nearest integer; the second fold consists of the next N/5 elements, etc, and the fifth fold consists of the remaining elements
# note: to create the folds consider: KFold(len(trainlabels), n_folds=5) from sklearn.cross_validation (http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html)
# note: once you have the folds use the rand_perm vector to get the random indices in the training data and labels
np.random.seed(0)
def knn(index, trainlabels, k, dis_matrix):
    results = hp.nsmallest(k, range(len(dis_matrix[index])), dis_matrix[index].__getitem__)
    total = 0
    for r in results:
        total += trainlabels[r]
    if total > np.floor(k/2):
        return 1
    else:
        return 0


#written by Taneli Virkkala

def cv(train, trainlabels, rand_perm):
    scaler = preprocessing.scale(train)
    dis_matrix = []
    err_matrix = []
    for i in range(len(scaler)):
        whole_line = []
        for j in range(len(scaler)):
            whole_line.append(np.linalg.norm(scaler[i]-scaler[j]))
        dis_matrix.append(whole_line)

    folds = cross_validation.KFold(len(trainlabels), n_folds=5)
    k = -1
    for x in range(13):
        temp_matrix1 = []
        k += 2
        for ftr, fte in folds:
            temp_matrix2 = []
            score = 0.0
            for y in fte:
                temp_value = knn(rand_perm[y], trainlabels, k, dis_matrix)
                #print y, temp_value
                temp_matrix2.append(temp_value)
            for z in range(len(fte)):
                if trainlabels[fte[z]] == temp_matrix2[z]:
                    score += 1
            temp_matrix1.append(float(score/20))
            #print temp_matrix1
        err_matrix.append(temp_matrix1)
        # print err_matrix

    max = 0
    new_k = 0
    for b in range(len(err_matrix)):
        scoore = 0
        for c in range(len(err_matrix[b])):
            scoore += err_matrix[b][c]
        if max < np.ceil(scoore/5):
            print max
            max = np.ceil(scoore/5)
            new_k = b*2 + 1

    #print max
    # print np.shape(err_matrix)
    # print len(err_matrix)
    new_matrix = np.zeros((5, 13))
    return new_k, new_matrix

data_train = np.loadtxt('parkinsonsTrain.dt')
labels = np.full(98, 0)
rander = np.random.permutation(len(labels))
cv(data_train, labels, rander)
