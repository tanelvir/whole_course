import numpy as np
import heapq as hp
from sklearn import ensemble
from sklearn import preprocessing
from numpy import genfromtxt
from sklearn import cross_validation


def cent_and_norm(train, test):
    tr = preprocessing.scale(train)
    scaler = preprocessing.StandardScaler().fit(train)
    te = scaler.transform(test)
    return tr, te

def knn(trainlabels, testlabels, k, dis_matrix): #inputs are: trainlabels, testlabels, the chosen k (1 e.g.) and already calculated distance matrix
    classes = [] #initialize vector for classifying
    for i in range(len(dis_matrix)): #go through every row in distance matrix
        results = hp.nsmallest(k, range(len(dis_matrix[i])), dis_matrix[i].__getitem__) #Heap structure is always sorted, and in this case by the distance. Last parameter means returning indeces of nearest neighbors.
        total = [] #for choosing the majority type
        for r in results: #going through indeces of nearest neighbors
            total.append(trainlabels[r]) #add label of that one neighbor
        counts = np.bincount(total) #calculates occurence of each label
        classes.append(np.argmax(counts)) #returns the most common label
    right = [] #vector for matches
    for j in range(len(classes)): #go through predictions
        if (testlabels[j] == classes[j]): #if it is a match
            right.append([1]) #increase by one

    print "K: " + str(k)
    print float(len(right))/float(len(classes)) #float number of accuracy rate
    return dis_matrix, classes #output: distance matrix and predicted classes

def knn2(i, trainlabels, k, dis_matrix): #modified knn with just given index
    results = hp.nsmallest(k, range(len(dis_matrix[i])), dis_matrix[i].__getitem__) #Heap structure is always sorted, and in this case by the distance. Last parameter means returning indeces of nearest neighbors.
    total = [] #for choosing the majority type
    for r in results: #going through indeces of nearest neighbors
        total.append(trainlabels[r]) #add label of that one neighbor
    counts = np.bincount(total) #calculates occurence of each label
    return np.argmax(counts) #returns the most common label

def multi_class(train_data, test_data):
    train = train_data[:,:54] #Leaving the last column out from train data
    test = test_data[:,:54] #Leaving the last column out from test data
    train_labels = train_data[:,54] #Creates label list for train
    test_labels = test_data[:,54] #Creates label list for test
    # train, test = cent_and_norm(train_data, test_data) #Activate if you want to center the data
    # for j in range(54): #Activate if you want to choose best amount for max_features
    classifier = ensemble.RandomForestClassifier(n_estimators=100, max_features=22, random_state=42) #creates classifier for forests
    classifier.fit(train, train_labels) #train the model
    results = classifier.predict(test) #predict test data
    accuracy = 0.0 #initial accuracy rate classificatio
    for i in range(len(results)): #count the right ones (same as 1 - error rate)
        if results[i] == test_labels[i]: #if it is a match
            accuracy += 1 #increase by one
    rate = float(accuracy/len(test_labels)) #float number of accuracy rate
    print str(rate) + " max_features:" + str(22)  #see accuracy rate with chosen max_features amount
    return results

#written by Taneli Virkkala

def cv(trainlabels, rand_perm, dis_matrix): #calculates cross-validation
    err_matrix = [] #initialize error matrix
    folds = cross_validation.KFold(len(trainlabels), n_folds=5) # 5 folds created by cross-validation
    k = -1 #initialize k
    for x in range(5): #k range 1-9
        temp_matrix1 = [] #matrix for each k
        k += 2 #k increase
        for ftr, tests in folds: #go through folds
            temp_matrix2 = [] #matrix/vector for each fold
            score = 0.0 #initialize accuracy rate
            for y in tests: #test folds through
                temp_value = knn2(rand_perm[y], trainlabels, k, dis_matrix) #label for one test cell
                temp_matrix2.append(temp_value) #add cell label
            for z in range(len(tests)): #all test cells through
                if trainlabels[tests[z]] == temp_matrix2[z]: #if it is a match
                    score += 1 #increase score
            temp_matrix1.append(float(score/len(tests))) #add score for that fold
        err_matrix.append(temp_matrix1) #assign that fold to k-value

    max = 0.0 #max score
    new_k = 0 #best k
    for b in range(len(err_matrix)): #go through error matrix
        scoore = 0 #initialize score
        for c in range(len(err_matrix[b])): #each k
            scoore += err_matrix[b][c] #each fold score
        print float(scoore/len(err_matrix[b])) #print value for that k
        if max <= float(scoore/len(err_matrix[b])): #if it is a new record
            max = float(scoore/len(err_matrix[b])) #assign a new max value
            new_k = b*2 + 1 #assign a new k

    print "this is max k: " + str(new_k) #print best k
    print "this is max value: " + str(max) #print best score
    return new_k, err_matrix


np.random.seed(0)
train_data = genfromtxt(open('C:/Users/Taneli/Downloads/data_analysis/final_exam/coverdata/covtype_train.csv','r'), delimiter=',', dtype='f8') #set your own path for forest cover train set
test_data = genfromtxt(open('C:/Users/Taneli/Downloads/data_analysis/final_exam/coverdata/covtype_test.csv','r'), delimiter=',', dtype='f8') #set your own path for forest cover test set
# multi_class(train_data, train_data) #classify train based on train data
# multi_class(train_data, test_data) #classify test based on train data
train = train_data[:,:54] #leaving the last column out from train data
test = test_data[:,:54] #leaving the last column out from test data
# train = train_data
# test = test_data
train, test = cent_and_norm(train, test) #center the data
train_labels = train_data[:, 54] #creates label list for train
test_labels = test_data[:, 54] #creates label list for test
k = 1 #initialize k
dis_matrix = [] #initialize distant matrix
for i in range(len(train)): #start doing train vs train distances
    whole_line = [] #per row (individual) neighbors
    for j in range(len(train)): #neighbors
        whole_line.append(np.linalg.norm(train[i] - train[j])) #Euclidean distance
    dis_matrix.append(whole_line) #line of distances to i's neighbors
# for o in range(5): #go through every k
#     knn(train_labels, train_labels, k, dis_matrix) #call knn function
#     k += 2 #increase k
rander = np.random.permutation(len(train_labels)) #creates random permutations for folds
cv(train_labels, rander, dis_matrix) #calls cross validation
k = 1 #initialize k
print "test"
dis_matrix = [] #initialize distant matrix
for i in range(len(test)): #start doing train vs test distances
    whole_line = [] #per row (individual) neighbors
    for j in range(len(test)): #neighbors
        whole_line.append(np.linalg.norm(test[i] - test[j])) #Euclidean distance
    dis_matrix.append(whole_line) #line of distances to i's neighbors
# # for o in range(5): #go through every k
# #     knn(train_labels, test_labels, k, dis_matrix) #call knn function
# #     k += 2 #increase k
rander = np.random.permutation(len(test_labels)) #creates random permutations for folds
cv(test_labels, rander, dis_matrix) #calls cross validation
