import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import sys
import tensorflow as tf
import csv
import time
import warnings
import multiprocessing as mp
from sklearn import preprocessing
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

min_max_scalle_x = preprocessing.MinMaxScaler((0,1))
min_max_scalle_y = preprocessing.MinMaxScaler((0,1))

def get_data(dataset, lat, lon):
    rg_lat      = lat
    rg_long     = lon
    latlon      = dataset.sel(latitude=rg_lat, longitude=rg_long, method='nearest')
    rg_rea      = pd.DataFrame({
        'time'      : pd.to_datetime(dataset.time.values),
        'Hs'        : latlon.swh.values,
        '10m-direc' : latlon.dwi.values,
        '10m-speed' : latlon.wind.values,
        'Period'    : latlon.pp1d.values
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
        epochs               = 2,
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

def create_future_predict(lead, df, npredict, num_features):
    if (lead==6):
        count = 0
    elif (lead==12):
        count = 6
    elif (lead==18):
        count = 12
    else:
        count = 18

    forecast                 = lead*2
    inputs                   = df['Hs'][:-(npredict+forecast+count)].values
    X, y                     = split_sequence(inputs, forecast, lead, False)
    X, y                     = prepare_data(X,y, False, num_features)
    
    model                    = get_model(1, forecast, num_features)
    history                  = compile_and_fit(model, X, y)

    x_input                  = df['Hs'][-(npredict+forecast+count):].values
    x_in, labels             = split_sequence(x_input, forecast, lead, False)
    x_in, labels             = prepare_data(x_in,labels, False, num_features)
    labels                   = min_max_scalle_y.inverse_transform(labels[:,0,0].reshape(-1, 1))
    
    result                   = pd.DataFrame()
    result['Data']           = df.index[-(npredict+count-lead+1):]
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

if __name__ == "__main__":
    rea         = xr.open_dataset(sys.argv[1])
    future      = sys.argv[2]
    rg_lat      = -31.53
    rg_long     = -49.86
    rg_rea      = get_data(rea, rg_lat, rg_long)
    
    inicio      = '2003-01-01'
    rg_rea      = rg_rea.loc[rg_rea.index >= inicio]
    
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
        num_features = 1
        npredict     = 744
    
        Parallel(n_jobs=1,backend='multiprocessing')(delayed(create_future_predict)(lead, rg_rea, npredict, num_features) for lead in lead_time)
    
        end                          = time.time()

    print('Time of execution: ',(end-start)/60, ' minutes.')
    print('###############################################')
    print('###############################################')
    print('##                                           ##')
    print('##      Simulation succesfully finished!     ##')
    print('##                                           ##')
    print('###############################################')
    print('###############################################')