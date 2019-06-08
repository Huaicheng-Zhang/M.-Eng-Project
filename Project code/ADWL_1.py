'''this files runs ADWL model on historical data, need to change 
the path of main file in input function'''

import numpy as np
import csv
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVR
from scipy.stats import norm, expon  , ks_2samp, kstest, t
from pandas import *
from sklearn.neighbors import KernelDensity
from sklearn.tree import DecisionTreeRegressor
import time

TrainingWindow = 15

#pre-processing function
def pre_processing(df):
    #code for extracting data for particular time
    hour = df.index.hour
    selector_new = (( 1 <= hour) & (hour <= 23))
    df1 = df[selector_new]



    #resampling in order to uniform the time series
    df2 = df1.resample('5min')

    #interpolation to fill missing values
    df3 = df2.interpolate(method ='cubic')


    return df3

#function to model the prediction error 
def error_model(data):


    #fitting t dist
    df, mu, std = t.fit(data, floc = 0)

    #fitting norm dist
    mu1, std1 = norm.fit(data, floc=0)

    print (' T dist parameteres are', a, mu, std)
    print ('Gaussian parameters are', mu1, std1)


    plt.figure()
    # Plot the histogram.
    plt.hist(data, bins=25, normed=True, alpha=0.5)

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = t.pdf(x, df, loc = mu, scale = std)
    p1 = norm.pdf(x, loc = mu1, scale = std1)
    kde_data = data[:, np.newaxis]
    X_plot = np.linspace(xmin, xmax, 1000)[:, np.newaxis]

    # Gaussian KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.035).fit(kde_data)
    log_dens = kde.score_samples(X_plot)
    #plt.plot(X_plot[:, 0], np.exp(log_dens), linewidth = 2)

    plt.plot(x, p, 'k', linewidth=2, label = 't-distribution')
    plt.plot(x, p1, 'r', linewidth=2, label = 'Gaussian distribution')
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    #plt.title(title)

    plt.xlabel('Normalized Error', fontsize = 16)
    plt.ylabel('Probability of Error', fontsize = 16)

    plt.legend()
    plt.show()


#main function
if __name__ == "__main__":
    start = time.clock()



    #creating DataFrame object
    df = pd.DataFrame()


    #reading csv file into DataFrame object
    dfa = pd.read_csv('C:/Users/THINKPAD/Desktop/M.Eng Pro/PredictiveAnalytics/03-2018.csv', parse_dates=['fecha'], index_col='fecha', sep = ';')

    #location selection: PM10005, PM10344, 18RV21PM01 and 03FL20PM01/ 90EL69PM01 42001 14XC71PM01
    #location could be changed according to its ID
    dfa = dfa[dfa['identif'] == 'PM10344']
    
    df_intensidad_1 = dfa[['vmed']]
    #('vmed' = velocity, 'intensidad' = 'intensity').

    #calling pre-processing function
    df_total = pre_processing(df_intensidad_1)


    #model implementation
    Y1= df_total.values

    std_Y1 = np.asarray(Y1)

    #regression method in the library needs array in specific format.
    #Converting input and output array in the required format
    X=[]
    Y=[]
    Y11 = []

    for i in range(len(df_total)):
        Y11.append(abs(Y1[i][0]))



    #getting rid of 0`s  (1~121 for speed, 1~1301 for intensity)
    for i in range(len(Y11)):
        if Y11[i] > 1 and Y11[i] < 1301:
            a=Y11[i]
            Y.append(a)





    for i in range(len(Y)):
        X.append([i])



    #initializes the model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    X_predicted = []
    Y_predicted = []


    #setting total length to iterate. 
    total = len(X)-TrainingWindow -2

    #iterations of i for every n, where n is window size. 3 in this case
    PredictionWindowArray =[]
    MAPE = []
    TrueError = []
    PredictionWindow = 2
    i=0

    #for s in range (0,total,1):
    while(i <= total):



    #
        X_training = X[i:i+TrainingWindow]
        Y_training = Y[i:i+TrainingWindow]
        X_testing = X[i+TrainingWindow:i+TrainingWindow+PredictionWindow]
        y_testing = Y[i+TrainingWindow:i+TrainingWindow+PredictionWindow]
        

        i = i + PredictionWindow

        #training and prediction

        y_rbf = svr_rbf.fit(X_training, Y_training).predict(X_testing)


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

    #taking the mean
        TempMAPE = (TempError1.mean())*100
        MAPE.append((TempMAPE))
        #it is predicting every value in a loop. all predicted values are appended in
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

    #calculating the prediction time
    print('processing time:',end - start)

    print ('error is', MAPEArray.mean())

    #Extracting data for testing and plotting
    Y_testing = Y[TrainingWindow:]
    X_testing = X[TrainingWindow:]


    #plot original/actual data
    plt.scatter(X_testing, Y_testing, c='k', label='Actual data')

    #plot predicted data
    plt.plot(X_predicted, Y_predicted, c='b', label = 'Proposed ADWL')
    
    #plt.scatter(X_predicted, Y_predicted, c='red', label = 'Predicted data')

    print ('the length of testing and prediction array is', len(Y_testing), len(Y_predicted))

    x_step =[]
    for i in range(len(PredictionWindowArray)):
        x_step.append(i)




    plt.xlim(0,200)

    plt.ylim(0,120)      # may change for different locations

    # The plot for traffic speed
    plt.xlabel('Time (min)')
    plt.ylabel('Average Traffic Speed (km/h)')
    plt.title('Traffic Speed')

    # The plot for traffic intensity
    # plt.xlabel('Time (min)')
    # plt.ylabel('Average Traffic Intensity (vehicles/h)')
    # plt.title('Traffic Intensity')
    plt.legend()


    plt.figure(2)
    plt.plot(PredictionWindowArray)


    plt.xlim(100,200)


    #converting the error in an array
    TrueError1 = np.asarray(TrueError)



    data = TrueError1

    #calling error modelling function
    error_model(TrueError1)

    plt.show()