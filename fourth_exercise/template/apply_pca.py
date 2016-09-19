#written by Taneli Virkkala

import numpy as np
import matplotlib.markers as mark
# input: Datamatrix as loaded by numpy.loadtxt('Irisdata.txt')
# output: Datamatrix of the projected data onto the two first principal components.

#Note this is based on model from stack overflow http://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
# Mark's answer given in answered Jan 13 '15 at 23:16. I have applied the model and not copied instantly

def covariance(data): #start calculate covariance
    length = np.shape(data)[1] #how many columns data has
    cov = np.empty((length, length)) #create covariance matrix
    for i in range(length): #go through each column
        cov[i, i] = np.mean(data[:,i] * data[:,i]) #add mean to diagonal
        for j in range(length): #go through neighbor cells
            cov[i,j] = cov[j,i] = np.mean(data[:,i] * data[:,j]) #add mean to neighbor cells
    return cov

def apply_pca(data): #calculate PCA
    data -= np.mean(data, 0) #zero the mean
    data /= np.std(data, 0) #zero the deviation
    Sigma = covariance(data) #calculate covariance matrix
    evals, evecs = np.linalg.eig(Sigma) #give eigenvalues and eigenvectors
    indexes = np.argsort(evals)[::-1][:2] #sort eigenvalues by reverse order
    Values, Vectors = evals[indexes], evecs[:, indexes] #take new sorted eigenvalues and eigenvectors
    print np.dot(Vectors.T, data.T).T #print the dot result
    return np.dot(Vectors.T, data.T).T #output the result

np.random.seed(0)
data = np.loadtxt('Irisdata.txt') #set your own path for Iris data
apply_pca(data) #run the program