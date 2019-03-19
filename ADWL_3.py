'''This files runs ADWL on historical data, you need to change 
the path of main dataset in input function'''

import numpy as np
import csv
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVR
from scipy.stats import norm, expon  , ks_2samp, kstest, t
from pandas import *
from sklearn.neighbors import KernelDensity
from sklearn.tree import DecisionTreeRegressor
import time
#from sklearn.utils import check_arrays

#define window sizes
TrainingWindow = 15


#main function
if __name__ == "__main__":
    start = time.clock()



    #creating DataFrame object
    df = pd.DataFrame()


    #reading csv file into DataFrame object
    dfa = pd.read_csv('C:/Users/THINKPAD/Desktop/M.Eng Pro/PredictiveAnalytics/09-2018Chic.csv', sep = ',')
    # location could be changed here according to ID
    dfa = dfa[dfa['linkId'] == 4616209]
    df_intensidad_1 = dfa[['Speed']]
    df_total_1 = df_intensidad_1
    
    df_total = df_total_1.interpolate(method ='cubic')

    #model implementation
    Y1= df_total.values

    # std_Y1 = np.asarray(Y1)
    # print "standard deviation of Input is", std_Y1.std()

    #regression method in the library needs array in specific format.
    #Converting input and output array in the required format
    X=[]
    Y=[]
    Y11 = []

    for i in range(len(df_total)):
        Y11.append(abs(Y1[i][0]))






    #getting rid of 0`s  (1~121 for speed, 1~1301 for intensity)
    for i in range(len(Y11)):
        if Y11[i] > 1 and Y11[i] < 121:
            a=Y11[i]
            Y.append(a)





    for i in range(len(Y)):
        X.append([i])


    #initialize the model and set relevant parameters
    svr_rbf = SVR(kernel='rbf', C=300, gamma=0.005, epsilon = 0.2)

    X_predicted = []
    Y_predicted = []


    #setting total length to iterate. 
    total = len(X)-TrainingWindow -2

    #iterations of i for every n, where n is window size. 3 in this case
    PredictionWindowArray =[]
    MAPE = []
    TrueError = []
    PredictionWindow = 1
    i=0

    #for s in range (0,total,1):
    while(i <= total):



    #
        X_training = X[i:i+TrainingWindow]
        Y_training = Y[i:i+TrainingWindow]
        X_testing = X[i+TrainingWindow:i+TrainingWindow+PredictionWindow]
        y_testing = Y[i+TrainingWindow:i+TrainingWindow+PredictionWindow]
        


        i = i + PredictionWindow

        #training and predicting

        y_rbf = svr_rbf.fit(X_training, Y_training).predict(X_testing)

        #a=y_rbf[0]
        #for j in y_rbf[0]:
        #Y_predicted.append(a)

        for j in y_rbf:
            Y_predicted.append(j)


        for k in X_testing:
            X_predicted.append(k)


        #calculating error (y_true - y_pred)/y_true
        TempError = []

        for m in range(len(y_testing)):
            a = abs((y_testing[m] - y_rbf[m])/y_testing[m])
            if a < 0.60:
                c = a
            b= ((y_testing[m] - y_rbf[m])/y_testing[m])
            if abs(b) < 0.60:
                d = b
            TempError.append(c)
            TrueError.append(d)

    #converting calculation results in an array
        TempError1 = np.asarray(TempError)

    #taking mean
        TempMAPE = (TempError1.mean())*100
        MAPE.append((TempMAPE))

        #it is predicting every value in a loop. all predicted values are added in
        #an array for overall error and graph


        #if error is greater then 10%, decrease the prediction window size
        if TempMAPE > 10:
            PredictionWindow = PredictionWindow -1

            if PredictionWindow == 0:
                PredictionWindow = 1
            PredictionWindowArray.append(PredictionWindow)
        #if error is less then 5%,increase the prediction window size
        if TempMAPE < 5:
            PredictionWindow = PredictionWindow +1

            if PredictionWindow > 3:
                PredictionWindow = PredictionWindow -1

            
            PredictionWindowArray.append(PredictionWindow)
        else:
            PredictionWindowArray.append(PredictionWindow)



    MAPEArray = np.asarray(MAPE)

    end = time.clock()
    print('processing time:',end - start)

    print ('error is', MAPEArray.mean())

    #Extracting data for testing and plotting
    Y_testing = Y[TrainingWindow:]
    X_testing = X[TrainingWindow:]


    #plot actual data
    plt.scatter(X_testing, Y_testing, c='k', label='Actual data')

    #plot predicted data
    plt.plot(X_predicted, Y_predicted, c='b', label = 'Proposed ADWL')

    #plt.scatter(X_predicted, Y_predicted, c='red', label = 'Predicted data')

    print ('the length of testing and prediction array is', len(Y_testing), len(Y_predicted))

    x_step =[]
    for i in range(len(PredictionWindowArray)):
        x_step.append(i)




    plt.xlim(0,200)

    plt.ylim(0,120)      # range may change for different locations

    plt.xlabel('Time (min)')
    plt.ylabel('Average Traffic Speed (km/h)')
    plt.title('Traffic Speed (Dataset 3)')
    
    # same plot for traffic inensity:
    # plt.xlabel('Time (min)')
    # plt.ylabel('Average Traffic Intensity (vehicles/h)')
    # plt.title('Traffic Intensity')
    plt.legend()


    plt.figure(3)
    plt.plot(X_testing, Y_testing)
    plt.xlim(0,200)

    plt.show()