from sklearn import preprocessing
import numpy as np
# input: 1) train: train data in the form of a N by d numpy array, where N is the number of training data points and d is the number of dimensions
#        2) test: test data in the form of a M by d numpy array, where M is the number of test data points and d is the number of dimensions    
# output: 1) centered and normalized train data as a numpy array
#         2) centered and normalized test data as a numpy array 



def cent_and_norm(train, test):
    tr = preprocessing.scale(train)
    scaler = preprocessing.StandardScaler().fit(train)
    te = scaler.transform(test)
    return tr, te