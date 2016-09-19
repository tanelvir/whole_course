#written by Taneli Virkkala

import numpy as np
import math
import functools
from sklearn import metrics
from sklearn import linear_model
from sklearn import datasets
# input:  1) train data (without labels) in the form of a N by d numpy array, where N is the number of training data points and d is the number of dimensions
#         2) trainlabels: labels for training data in the form of a N by 1 numpy vector, where N is the number of training data points
#         3) test: test data (without labels) in the form of a M by d numpy array, where M is the number of test data points and d is the number of dimensions
#
# output: 1) vector (numpy array) consisting of the predicted classes for the test data
#         2) the beta parameter of the model
# note: the labels should **not** be part of the train/test data matrices!

#Just run the whole file
def logreg(train_data, train_labels, test_data, test_labels):
    log_reg = linear_model.LogisticRegression() #creates the classifier
    log_reg.fit(train_data, train_labels) #train the model
    predicted = log_reg.predict(test_data) #predict test data
    accuracy_rate = 0.0 #initial accuracy rate classification
    for p in range(len(predicted)): #count the right ones (same as 1 - error rate)
        if predicted[p] == test_labels[p]: #if it is a match
            accuracy_rate += 1 #increase by one
    print float(accuracy_rate/len(test_labels)) #float number of accuracy rate
    #print log_reg.coef_ #print beta
    return predicted, log_reg.coef_ #output: classified elements in a vector and beta

np.random.seed(0)
train_data = np.loadtxt('C:/Users/Taneli/Downloads/data_analysis/final_exam/redwine_train.txt') #set your own path for redwine train set
train_labels = np.loadtxt('C:/Users/Taneli/Downloads/data_analysis/final_exam/redwinedata/redwine_trainlabels.txt') #set your own path for redwine trainlabels
test_data = np.loadtxt('C:/Users/Taneli/Downloads/data_analysis/final_exam/redwinedata/redwine_test.txt') #set your own path for redwine test set
test_labels = np.loadtxt('C:/Users/Taneli/Downloads/data_analysis/final_exam/redwinedata/redwine_testlabels.txt') #set your own path for redwine testlabels
logreg(train_data, train_labels, train_data, train_labels) #Run the code to see predictions on train set
logreg(train_data, train_labels, test_data, test_labels) #Run the code to see predictions on test set