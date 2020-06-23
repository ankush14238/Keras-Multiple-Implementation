import os
import keras
import tensorflow as tf
from keras.layers import Input, Dense, Activation
from keras.models import Model
import keras.optimizers
from keras.callbacks import EarlyStopping
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


            
def r_sq(X, yhat, y):
    yhat = np.squeeze(yhat)
    y = np.squeeze(y)
    SS_Residual = sum((y-yhat)**2)
    SS_Total = sum((y-np.mean(y))**2)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
    return r_squared, adjusted_r_squared




def nn(X_train_mat, X_test_mat, y_train_vec, y_test_vec, optimizer_used, activation_used, network):
    train_rows, train_cols = X_train_mat.shape
    print(X_train_mat.shape)
    inputs = Input(shape=(train_cols, ))
    
    if network == 'perceptron':
        output = Dense(1, activation='linear')(inputs)

    elif network == 'nn3l': 
        x = Dense(5, activation=activation_used)(inputs)
        output = Dense(1, activation='linear')(x)

    elif network == 'nnxl':
        x = Dense(5, activation=activation_used)(inputs)
        x = Dense(5, activation=activation_used)(x)
        output = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=output)
    if optimizer_used == 'sgd':
        opt = keras.optimizers.SGD()
    elif optimizer_used == 'sgd_mom':
        opt = keras.optimizers.SGD(momentum=0.1)
    elif optimizer_used == 'adam':
        opt = keras.optimizers.Adam()

    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['mse', 'mae'])
    es = EarlyStopping(monitor='val_loss', patience=10)
    training = model.fit(X_train_mat, y_train_vec, verbose = 0, batch_size=min(128, train_rows), epochs=2000, callbacks=[es], validation_data=(X_test_mat, y_test_vec))
    num_epochs = len(training.history['loss'])
    y_train_pred = model.predict(X_train_mat)
    y_test_pred = model.predict(X_test_mat)
    rSq, rSqBar = r_sq(X_train_mat, y_train_pred, y_train_vec)
    rSqCv, rSqBarCv = r_sq(X_train_mat, y_test_pred, y_test_vec)
    
    return rSq, rSqBar, rSqCv, num_epochs

    
def forward(dataset, optimizer_used, activation_used, network, output_list):
    data = pd.read_csv(dataset)
    rows, cols = data.shape
    X = data.iloc[:,:cols-1]
    y = data.iloc[:,cols-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    R_sq_list = [0 for x in range(cols-1)]
    R_sq_bar_list = [0 for x in range(cols-1)]
    R_sq_cv_list = [0 for x in range(cols-1)]
    epochs_list = [0 for x in range(cols-1)]
    x_selected = []
    x_remaining = [int(x) for x in range(cols-1)]    
    for i in range(cols-1):

        r_max = -9999999
        picked_column = -1
        for j in x_remaining:
            current = x_selected + [j]
            results = nn(X_train.iloc[:,current], X_test.iloc[:,current], y_train, y_test, optimizer_used, activation_used, network)
            r = results[0]

            
            if r > r_max:
                r_max = r 
                picked_column = j
                R_sq_list[i] = max(0, results[0] * 100)
                R_sq_bar_list[i] = max(0, results[1] * 100)
                R_sq_cv_list[i] = max(0, results[2] * 100)
                epochs_list[i] = results[3]
        if picked_column == -1:
            break
        x_selected.append(picked_column)
        x_remaining.remove(picked_column)
        
    x_axis = [int(x) + 1 for x in range(len(R_sq_list))]

    plt.plot(x_axis, R_sq_list, '--ro', label = 'rSq')
    plt.plot(x_axis, R_sq_bar_list, '--bs', label = 'rSqBar')
    plt.plot(x_axis, R_sq_cv_list, '--g^', label = 'rSqCv')
    plt.ylabel('R Square values')
    plt.xlabel('Number of parameters')
    plt.legend(loc='upper left')
    plt.savefig('%s_%s_%s_%s.png' %(dataset, optimizer_used, activation_used, network))
    plt.clf()
    epochs_list[:] = (value for value in epochs_list if value != 0)
    average_epochs = int(sum(epochs_list) / max(len(epochs_list), 1))
    n_opt_rsqbar = R_sq_bar_list.index(max(R_sq_bar_list)) + 1
    n_opt_rsqcv = R_sq_cv_list.index(max(R_sq_cv_list)) + 1
    sentance = '%s_%s_%s_%s Avg_epochs = %d, n*_rSqBar = %d, n*_rSqCv = %d' %(dataset, optimizer_used, activation_used, network, average_epochs, n_opt_rsqbar, n_opt_rsqcv)
    output_list.append(sentance)
    print(sentance)
        
datasets = ['data.csv']
num_datasets = len(datasets)

all_optimizers = ['adam']
all_activations = ['linear']
all_networks = ['perceptron', 'nn3l', 'nnxl']


output_list = []
for optimizer in all_optimizers:
    for activation in all_activations:
        for dataset in datasets:
            for network in all_networks:
                forward(dataset, optimizer, activation, network, output_list)
                print(output_list)
                

with open('stats.txt', 'w') as f:
    for item in output_list:
        f.write("%s\n" % item)