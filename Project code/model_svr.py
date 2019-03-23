'''This program shows the performance of SVR algorithm and related prediction error'''

import numpy as np
import csv
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVR
from scipy.stats import norm, expon, ks_2samp, kstest, t
from pandas import *
from sklearn.neighbors import KernelDensity
from sklearn import datasets, linear_model
import time

#define window sizes
TrainingWindow = 20
PredictionWindow = 1

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


#creating DataFrame object
df = pd.DataFrame()

#reading csv file into DataFrame object （Dataset 1-3）

dfa = pd.read_csv('C:/Users/THINKPAD/Desktop/M.Eng Pro/PredictiveAnalytics/03-2018.csv', parse_dates=['fecha'], index_col='fecha', sep = ';')
dfa = dfa[dfa['identif'] == 'PM10344']
df_intensidad_1 = dfa[['vmed']]   #feature. 'intensidad' or 'vmed'
df_total = pre_processing(df_intensidad_1)


# dfa = pd.read_csv('C:/Users/THINKPAD/Desktop/M.Eng Pro/PredictiveAnalytics/08-2016.csv', parse_dates=[2], index_col='time_id', sep = ',')
# dfa = dfa[dfa['road_id'] == 37]
# df_intensidad_1 = dfa[['speed']]
# df_total_1 = df_intensidad_1

# dfa = pd.read_csv('C:/Users/THINKPAD/Desktop/M.Eng Pro/PredictiveAnalytics/09-2018Chic.csv', sep = ',')
# dfa = dfa[dfa['linkId'] == 4616209]
# df_intensidad_1 = dfa[['Speed']]
# df_total_1 = df_intensidad_1













Y1= df_total.values




#regression method in the library needs array in specific format.
#Converting input and output array in the required format

X=[]
Y=[]
Y11 = []

for i in range(len(df_total)):
    Y11.append(abs(Y1[i][0]))



#getting rid of 0`s and outliers
for i in range(len(Y11)):
    if Y11[i] > 1 and Y11[i] < 131:
        a=Y11[i]
        Y.append(a)



for i in range(len(Y)):
    X.append([i])





print (len(X), len(Y))
#print (X, Y)
X_predicted = []
Y_predicted = []









#initializing different models
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)



MAPE = []
total = 2000-TrainingWindow -2
i = 0
while(i <= total):
    X_training = X[i:i+TrainingWindow]
    Y_training = Y[i:i+TrainingWindow]

    X_testing = X[i+TrainingWindow:i+TrainingWindow+PredictionWindow]
    Y_testing = Y[i+TrainingWindow:i+TrainingWindow+PredictionWindow]

    i = i + PredictionWindow

    y_rbf = svr_rbf.fit(X_training, Y_training)

    e = y_rbf.predict(X_testing)



    for j in e:
        Y_predicted.append(j)


    for k in X_testing:
        X_predicted.append(k)


    TempError = []
    for m in range(len(Y_testing)):
        a = abs((Y_testing[m] - e[m])/Y_testing[m])
        # a = (Y_testing[m] - e[m])**2/len(Y_testing)

        if a < 1:

            c = a
        else:
            c = 0
        if c > 0:
            TempError.append(c)

    TempError1 = np.asarray(TempError)
    TempMAPE = (TempError1.mean())*100
    MAPE.append((TempMAPE))

MAPEArray = np.asarray(MAPE)
print ('error is', MAPEArray.mean())

MAPE = []
X_predicted1 = []
Y_predicted1 = []
PredictionWindow = 30
total = 2000-TrainingWindow -2
i = 0
while(i <= total):
    X_training = X[i:i+TrainingWindow]
    Y_training = Y[i:i+TrainingWindow]

    X_testing = X[i+TrainingWindow:i+TrainingWindow+PredictionWindow]
    Y_testing = Y[i+TrainingWindow:i+TrainingWindow+PredictionWindow]

    i = i + PredictionWindow

    y_rbf = svr_rbf.fit(X_training, Y_training)

    e = y_rbf.predict(X_testing)



    for j in e:
        Y_predicted1.append(j)


    for k in X_testing:
        X_predicted1.append(k)


    TempError = []
    for m in range(len(Y_testing)):
        a = abs((Y_testing[m] - e[m])/Y_testing[m])
        # a = (Y_testing[m] - e[m])**2/len(Y_testing)

        if a < 1:

            c = a
        else:
            c = 0
        if c > 0:
            TempError.append(c)

    TempError1 = np.asarray(TempError)
    TempMAPE = (TempError1.mean())*100
    MAPE.append((TempMAPE))

MAPEArray = np.asarray(MAPE)
print ('error is', MAPEArray.mean())


#Extracting data for testing and plotting
Y_testing = Y[TrainingWindow:]
X_testing = X[TrainingWindow:]

#plot original/actual data
plt.scatter(X_testing, Y_testing, c='k', label='Actual data')

#plot predicted data
plt.plot(X_predicted, Y_predicted, c='b', label = 'prediction window = 1')
plt.plot(X_predicted1, Y_predicted1, c='g', label = 'prediction window = 30')

# x_step =[]
# for i in range(PredictionWindow):
#     x_step.append(i)

plt.xlim(0,200)

plt.ylim(0,120)      # may change for different locations

plt.xlabel('Time (min)')
plt.ylabel('Average Traffic Speed (km/h)')
plt.title('Traffic Speed')
plt.legend()
plt.show()






