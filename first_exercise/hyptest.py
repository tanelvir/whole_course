import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return True if the null hypothesis is accepted and False otherwise
# def hyptest(data):
#     non_smoker_list = []
#     smoker_list = []
#     for i in range(len(data)):
#         if data[i,4] == 0.0:
#             non_smoker_list.append(data[i,1])
#         else:
#             smoker_list.append(data[i,1])
#     non_mean = np.mean(non_smoker_list)
#     smoker_mean = np.mean(smoker_list)
#     non_sd = np.std(non_smoker_list)
#     smoker_sd = np.std(smoker_list)
#     ti1 = (smoker_mean-non_mean)
#     ti2 = np.sqrt(((smoker_sd**2)/len(smoker_list))+((non_sd**2)/len(non_smoker_list)))
#     ti = ti1/ti2
#     print ti
#     v_upper = (((smoker_sd**2)/len(smoker_list))+((non_sd**2)/len(non_smoker_list)))**2
#     v_lower_left = (smoker_sd**4)/(len(smoker_list)**2*(len(smoker_list)-1))
#     v_lower_right = (non_sd**4)/(len(non_smoker_list)**2*(len(non_smoker_list)-1))
#     v = np.floor(v_upper/(v_lower_left + v_lower_right))
#     print v
#     t_value = 2*t.cdf(-ti, v)
#     print t_value
#     significance_value = 0.05
#     if t_value > significance_value:
#         return 0
#     else:
#         return 1

# data = np.loadtxt('C:\Users\Taneli\Downloads\data_analysis\smoking.txt')
# print hyptest(data)

#Based on my uploaded code to code checker from assignment 1, exercise 3

#Just run the whole file












#written by Taneli Virkkala

def hyptest(data, labels):
    good = []  #good wines array
    bad = []   #bad wines array
    for i in range(len(labels)):    #dividing into two groups: bad and good wines
        if labels[i] == 1.0:        #if wine is good
            good.append(data[i, 10]) #adding good wines
        else:
            bad.append(data[i, 10])
    good_mean = np.mean(good)
    bad_mean = np.mean(bad)
    good_sd = np.std(good)
    bad_sd = np.std(bad)
    pi1 = (bad_mean - good_mean) #(X - Y) numerator of t equasion
    pi2 = np.sqrt(((bad_sd ** 2) / len(bad)) + ((good_sd ** 2) / len(good))) #denominator of t equasion
    p_value = pi1 / pi2 #whole p-value (t) based on assignment 1, page 2 formula
    print p_value
    v_upper = (((bad_sd**2)/len(bad))+((good_sd**2)/len(good)))**2 #whole degrees
    v_lower_left = (bad_sd**4)/(len(bad)**2*(len(bad)-1))          #of freedom (v)
    v_lower_right = (good_sd**4)/(len(good)**2*(len(good)-1))      #calculation based on assignment 1, page 2 formula
    v = np.floor(v_upper/(v_lower_left + v_lower_right)) #degrees of freedom for calculating t-statistic. One value is the output.
    print v #degrees of freedom
    t_value = 2*t.cdf(-p_value, v) #t-statistic based on assignment 1, page 2 formula
    print t_value
    significance_value = 0.05 #measuring if nyll hypothesis is accepted
    if t_value > significance_value:
        return 0 #null hypothesis accepted
    else:
        return 1 #null hypothesis rejected

data = np.loadtxt('C:/Users/Taneli/Downloads/data_analysis/final_exam/redwine_train.txt') #set your own path for redwine train set
labels = np.loadtxt('C:/Users/Taneli/Downloads/data_analysis/final_exam/redwinedata/redwine_trainlabels.txt') #set your own path for redwine trainlabels set
print hyptest(data, labels) #see the output















