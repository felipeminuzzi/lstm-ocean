import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import sys
import tensorflow as tf
import csv
import time
import warnings
import multiprocessing as mp
from hampel import hampel
from sklearn import preprocessing
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

import seaborn as sns
from numpy import percentile
from pyod.models.knn import KNN
from scipy import stats

'''
previsao com boia historica e remocao dos outliers
'''


min_max_scalle_x = preprocessing.MinMaxScaler((0,1))
min_max_scalle_y = preprocessing.MinMaxScaler((0,1))

def get_data(df):
    rg_rea      = pd.DataFrame({
        'time'      : df['# Datetime'].values,
        'Hs'        : df['Wvht'].values,
    })
    rg_rea      = rg_rea.set_index('time')
    return rg_rea

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

def erro(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred)/y_true)*100

def split_sequence(sequence, n_steps_in, lead_time, flag):
    X, y = [], []
    m = len(sequence) - lead_time
    for i in range(m):
        end_ix     = i + n_steps_in
        out_end_ix = end_ix + int(n_steps_in*2)
        if flag == True:
            if out_end_ix > m:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        else:
            if end_ix > m:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix+lead_time-1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y) 

def compile_and_fit(model, X, y, patience=20):
    early_stopping           = tf.keras.callbacks.EarlyStopping(
        monitor              = 'val_loss',
        patience             = patience,
        mode                 = 'min'
    )
    model.compile(
        loss                 = 'mean_absolute_error',
        optimizer            = 'adam',
        metrics              = ['mean_squared_error']
    )
    history                  = model.fit(
        X,
        y,
        epochs               = 500,
        validation_split     = 0.2,
        callbacks            = [early_stopping],
        verbose              = 0
    )
    return history

def prediction(model, data_in, n_steps_in, num_features):
    x_predict = data_in.reshape(1, n_steps_in, num_features)
    predict                      = model.predict(x_predict)
    predict                      = min_max_scalle_x.inverse_transform(predict[0,0].reshape(-1, 1))
    return predict[0,0]

def prepare_data(X,y, flag, num_features):
    dim_1                        = X.shape[0]
    dim_2                        = X.shape[1]
    if flag:
        dim_3                        = y.shape[1]
    X                            = X.flatten()
    y                            = y.flatten()
    X                            = min_max_scalle_x.fit_transform(X.reshape(-1, 1))
    y                            = min_max_scalle_y.fit_transform(y.reshape(-1, 1))
    X                            = X.reshape((dim_1, dim_2, num_features))
    if flag:
        y                            = y.reshape((dim_1, dim_3, num_features))
    else:
        y                            = y.reshape((dim_1, 1, num_features))
    return X,y

def plot(df,error,lead, mape_value):
    plt.figure(1)
    plt.plot(df['Data'], df['Hs Reanalysis Value'], 'r-', label='Label (dado real)')
    plt.plot(df['Data'], df['Hs Predict Value'], 'b-', label='Previsão')
    plt.xlabel('Data')
    plt.ylabel('Wave height (Hs)')
    plt.legend()
    plt.title(f'Previsão de altura de onda para lead time {lead}. MAPE: {mape_value}')
    plt.savefig(f'lstm_predicts_{lead}_leadtime.png', bbox_inches='tight')

    plt.figure(2)
    plt.plot(df['Data'], error, 'r-')
    plt.xlabel('Data')
    plt.ylabel('Absolute error')
    plt.title(f'Erro absoluto para lead time {lead}. MAPE: {mape_value}')
    plt.savefig(f'lstm_erro_{lead}_leadtime.png', bbox_inches='tight')

def plot_future(df,mape_value):
    plt.figure(1)
    plt.plot(df['Data'], df['Hs Predict Value'], 'r-', label='Previsão')
    plt.xlabel('Data')
    plt.ylabel('Wave height (Hs)')
    plt.legend()
    plt.title(f'Previsão de altura de onda futuro. MAPE da validação: {mape_value}')
    plt.savefig(f'lstm_predicts_future.png', bbox_inches='tight')

def get_model(num_prev, forecast, num_features):
    modelo             = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(forecast, num_features)),
        tf.keras.layers.LSTM(48, activation='relu', return_sequences=True, input_shape=(forecast, num_features)),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=False),
        tf.keras.layers.Dense(num_prev, kernel_initializer=tf.initializers.zeros)])
    return modelo

def create_future_predict(lead, df, npredict, forecast, num_features):
    if (lead==6):
        count = 0
    elif (lead==12):
        count = 6
    elif (lead==18):
        count = 12
    else:
        count = 18
    inputs                   = df['Hs'][:-(npredict+count)].values
    X, y                     = split_sequence(inputs, forecast, lead, False)
    X, y                     = prepare_data(X,y, False, num_features)
    
    model                    = get_model(1, forecast, num_features)
    history                  = compile_and_fit(model, X, y)

    x_input                  = df['Hs'][-(npredict+count):].values
    x_in, labels             = split_sequence(x_input, forecast, lead, False)
    x_in, labels             = prepare_data(x_in,labels, False, num_features)
    labels                   = min_max_scalle_y.inverse_transform(labels[:,0,0].reshape(-1, 1))
    
    result                   = pd.DataFrame()
    result['Data']           = df.index[-npredict+forecast+5:]
    result['Hs Reanalysis Value'] = labels[:,0]
    
    predictions              = []
    predictions              = [prediction(model, dado, forecast, num_features) for dado in x_in]
    result['Hs Predict Value'] = predictions
    
    mape_model           = mape(result['Hs Reanalysis Value'], result['Hs Predict Value'])
    error                = erro(result['Hs Reanalysis Value'], result['Hs Predict Value'])
    
    result.to_csv(f'lstm_predictions_{lead}_leadtime.csv')
    plot(result,error,lead,mape_model) 

def create_non_lead_future(lead, df, npredict, forecast, train_size, num_features):
    inputs                   = df['Hs'][:-train_size].values
    X, y                     = split_sequence(inputs, forecast, lead, True)
    X, y                     = prepare_data(X,y, True)
    
    model                    = get_model(npredict)
    history                  = compile_and_fit(model, X, y)
    mape_val                 = history.history['val_mape'][-1]

    x_input                  = df['Hs'][-train_size:].values
    x_in                     = min_max_scalle_x.fit_transform(x_input.reshape(-1, 1))
    x_predict                = x_in.reshape(1, len(x_in), num_features)
    
    predict                  = model.predict(x_predict)
    predict                  = min_max_scalle_x.inverse_transform(predict[0,:].reshape(-1, 1))
    
    result                   = pd.DataFrame()
    result['Data']           = pd.date_range(start=df.index[-1], periods=npredict+1, freq='H')[1:]
    result['Hs Reanalysis Value'] = np.nan
    result['Hs Predict Value'] = predict[:,0]

    result.to_csv(f'lstm_predictions_future.csv')
    plot_future(result,mape_val) 

def knn_filter(buoy_filepath, cols):    

    mpl.rcParams["figure.dpi"] = 200

    wave = pd.read_csv(buoy_filepath, sep=',')
    
    wave[wave[cols[0]] == -9999] = np.nan
    wave[wave[cols[1]] == -9999] = np.nan
    wave = wave.dropna()

    raw = pd.read_csv(buoy_filepath, sep=',')
    raw[raw[cols[0]] == -9999] = np.nan
    raw[raw[cols[1]] == -9999] = np.nan
    raw = raw.dropna()

    minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
    wave[cols] = minmax.fit_transform(wave[cols])

    X1 = wave[cols[1]].values.reshape(-1, 1)
    X2 = wave[cols[0]].values.reshape(-1, 1)

    X = np.concatenate((X1, X2), axis=1)

    outliers_fraction = 0.01

    classifiers = {"K Nearest Neighbors (KNN)": KNN(contamination=outliers_fraction)}

    xx , yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    outliers = {}

    for i, (clf_name, clf) in enumerate(classifiers.items()):
        clf.fit(X)

        # predict raw anomaly score
        scores_pred = clf.decision_function(X) * -1

        # prediction of a datapoint category outlier or inlier
        y_pred = clf.predict(X)
        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)
        plt.figure(figsize=(7, 7))

        # copy of dataframe
        df = wave.copy()
        df['outlier'] = y_pred.tolist()

        # creating a combined dataframe of outliers from the 4 models
        outliers[clf_name] = df.loc[df['outlier'] == 1]

        # IN1 - inlier feature 1,  IN2 - inlier feature 2
        IN1 =  np.array(df[cols[1]][df['outlier'] == 0]).reshape(-1,1)
        IN2 =  np.array(df[cols[0]][df['outlier'] == 0]).reshape(-1,1)


        # OUT1 - outlier feature 1, OUT2 - outlier feature 2
        OUT1 =  df[cols[1]][df['outlier'] == 1].values.reshape(-1,1)
        OUT2 =  df[cols[0]][df['outlier'] == 1].values.reshape(-1,1)

        print('OUTLIERS:',n_outliers, '|', 'INLIERS:',n_inliers, '|', 'MODEL:',clf_name)

            # threshold value to consider a datapoint inlier or outlier
        threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)

        # decision function calculates the raw anomaly score for every point
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        Z = Z.reshape(xx.shape)

        # fill blue map colormap from minimum anomaly score to threshold value
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.GnBu_r)

        # draw red contour line where anomaly score is equal to thresold
        a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')

        # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
        plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='lemonchiffon')

        b = plt.scatter(IN1,IN2, c='white',s=20, edgecolor='k')

        c = plt.scatter(OUT1,OUT2, c='black',s=20, edgecolor='k')

        plt.axis('tight')  

        # loc=2 is used for the top left corner 
        plt.legend(
            [a.collections[0], b,c],
            ['Decision function', 'Inliers','Outliers'],
            prop=mpl.font_manager.FontProperties(size=13),
            loc=2)

        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
                    labels=np.round(np.linspace(round(raw[cols[0]].min()), 
                                        round(raw[cols[0]].max()), 
                                        6), 1))
        plt.xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
                    labels=np.round(np.linspace(round(raw[cols[1]].min()), 
                                        round(raw[cols[1]].max()), 
                                        6), 1))
        plt.xlabel('Tp (s)')
        plt.ylabel('Hs (m)')
        plt.title(clf_name)
        plt.savefig(f'outliers.png', bbox_inches='tight')

        raw['outlier'] = df['outlier']
        filtered = raw[raw['outlier'] != 1]
        
        return filtered.reset_index(drop=True)


if __name__ == "__main__":
    rea         = sys.argv[1]
    future      = sys.argv[2]
    cols        = ['Wvht', 'Dpd']
    rg_rea      = knn_filter(rea, cols)
    rg_rea      = get_data(rg_rea)
            
    start                        = time.time()
    if future.lower() == 'future':
        npredict    = 24*7
        train_size   = int(rg_rea.shape[0]*0.2)
        forecast     = int(npredict/2)
        num_features = 1
        start                        = time.time()
        create_non_lead_future(0, rg_rea, npredict, forecast, train_size, num_features)
        end                          = time.time()
    else:
        lead_time    = [6,12,18,24]
        forecast     = 12
        num_features = 1
        npredict     = 744
    
        Parallel(n_jobs=-1,backend='multiprocessing')(delayed(create_future_predict)(lead, rg_rea, npredict, forecast, num_features) for lead in lead_time)
    
        end                          = time.time()

    print('Time of execution: ',(end-start)/60, ' minutes.')
    print('###############################################')
    print('###############################################')
    print('##                                           ##')
    print('##      Simulation succesfully finished!     ##')
    print('##                                           ##')
    print('###############################################')
    print('###############################################')