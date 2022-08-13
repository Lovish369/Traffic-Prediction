import time
st_lib = time.time()
import numpy as np
import pandas as pd
from glob import glob
from keras.metrics import MeanSquaredError as mae
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import train_test_split
from math import sqrt
import datetime
import tensorflow as tf
from keras.models import load_model
import keras
from tcn import TCN
from bokeh.plotting import figure,output_file, show
import os, psutil
et_lib = time.time()
elp_lib = et_lib - st_lib
print('Importing lib in:', elp_lib, ' sec')


st_code = time.time()
filenames = glob('/home/pi/Desktop/Lovish/TCN6/402214'+'/*.csv')
#print(filenames)

data = []
for filename in filenames:
    data.append(pd.read_csv(filename))
full_data =pd.concat(data,ignore_index=True)

cols = list(full_data)[2]
training_data=full_data[cols].astype(str)
training_data = np.array(training_data).reshape(len(training_data),1)
scaler =MinMaxScaler(feature_range=(0,1))
training_scaled = scaler.fit_transform(training_data)

#print(training_scaled[:15],training_scaled[15,-1])

n_steps = 15
def split_sequences(sequences):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def split_set1(X):
    S = []
    HS= []
    i=0
    while i < len(X)-n_steps:
        l=[]
        l.append(X[i:n_steps+i])
        S.append(l)
        l=[]
        l.append(X[n_steps+i])
        HS.append(l)
        i+=1
    S=np.array(S)
    HS=np.array(HS)
    S=S.reshape(S.shape[0],S.shape[2],S.shape[3])
    HS=HS.reshape(HS.shape[0],HS.shape[1])
    return S,HS

X,Y = split_set1(training_scaled)
t_d=len(training_scaled)//5
#print(t_d)

train_X,test_X,train_Y,test_Y = X[:-t_d],X[-t_d:-1],Y[:-t_d],Y[-t_d:-1]
#print(len(test_X))

test_Y=test_Y.reshape(test_Y.shape[0],1)

#print(train_Y.shape)
#print(train_X.shape)

model = load_model('/home/pi/Desktop/Lovish/TCN6/model_tcn')

st_predict = time.time()
y_predict_scaled =model.predict(test_X)
stop_predict = time.time()
tt_predict = stop_predict - st_predict
print('Prediction Time: ', tt_predict, 'sec')

y_predict_scaled
#print(y_predict_scaled.shape)

y_predict = scaler.inverse_transform(y_predict_scaled)
test_Y = scaler.inverse_transform(test_Y)

print('MAE: ',mae(y_predict,test_Y))
print('RMSE: ',sqrt(mse(y_predict,test_Y)))
print('MAPE: ',mape(y_predict,test_Y))

predict=list(y_predict)[52:288+52]
#print(predict)
st_code_end = time.time()
elp_code = st_code_end - st_code
print('Code takes:', elp_code, ' sec to run')

process = psutil.Process(os.getpid())
print('Program takes',process.memory_info().rss / 1024 ** 2, 'mega-bytes of RAM')


"""
# Plot input data  and predicted outp
p = figure(x_axis_type = 'datetime',title="Traffic Flow", x_axis_label='Time', y_axis_label='Veh/5min',plot_width=1200, plot_height=800)

output_file("/home/pi/Desktop/Lovish/TCN6/TCN_Traffic flow.html")

# # #For graphing using Bokeh
timeAxis = [datetime.timedelta(minutes=5*i) for i in range(0,288)]
pred1= test_Y
pred2 = list(pred1)[52:288+52]

p.line(timeAxis, pred2, color='green', legend_label='Expected Traffic Flow TCN ',line_width = 2)
p.line(timeAxis,predict, color='red', legend_label='Predicted Traffic Flow TCN ',line_width = 2)

# # show the results
show(p)
"""

