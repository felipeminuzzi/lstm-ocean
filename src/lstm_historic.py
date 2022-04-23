import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import sys
import os
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
        metrics              = ['mean_squared_error','mape']
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
        dim_3                    = y.shape[1]
    X                            = X.flatten()
    y                            = y.flatten()
    X                            = min_max_scalle_x.fit_transform(X.reshape(-1, 1))
    y                            = min_max_scalle_y.fit_transform(y.reshape(-1, 1))
    X                            = X.reshape((dim_1, dim_2, num_features))
    if flag:
        y                        = y.reshape((dim_1, dim_3, num_features))
    else:
        y                        = y.reshape((dim_1, 1, num_features))
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

def create_future_predict(lead, df, npredict, forecast, num_features, inicio):
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
    
    mape_model               = mape(result['Hs Reanalysis Value'], result['Hs Predict Value'])
    cols                     = ['Data inicio','Lead time', 'MAPE']
    mapes_results            = pd.DataFrame([[inicio] + [lead] +[mape_model]], columns=cols)

    result.to_csv(f'lstm_predictions_{lead}_{inicio}_leadtime.csv')

    return mapes_results
     

def create_non_lead_future(lead, df, npredict, forecast, train_size, num_features):
    inputs                   = df['Hs'][:-train_size].values
    X, y                     = split_sequence(inputs, forecast, lead, True)
    X, y                     = prepare_data(X,y, True, num_features)
    
    model                    = get_model(npredict, forecast, num_features)
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

def predict_historic(df, inicio, path, future):
    df_inicio = df.loc[df.index >= inicio]

    if future.lower() == 'future':
        npredict    = 24*7
        train_size   = int(df.shape[0]*0.2)
        forecast     = int(npredict/2)
        num_features = 1
        start                        = time.time()
        create_non_lead_future(0, df, npredict, forecast, train_size, num_features)
        end                          = time.time()
        print(f'Time of execution: ,{(end-start)/60}, minutes for lead {lead}')
    else:
        lead_time    = [6,12,18,24]
        forecast     = 12
        num_features = 1
        npredict     = 744
        
        df_final     = pd.DataFrame()
        start                        = time.time()
        for lead in lead_time:
            dataframe = create_future_predict(lead, df_inicio, npredict, forecast, num_features, inicio)
            df_final  = df_final.append(dataframe, ignore_index=True)
        df_final.to_csv(f'{path}lstm_historic_mapes_{inicio}.csv')

        end                          = time.time()
        print(f'Time of execution: ,{(end-start)/60}, minutes for lead {lead} and initial data {inicio}')
        
if __name__ == "__main__":
    inicios_datasets = [
    '1979-01-01','1980-01-01','1981-01-01','1982-01-01','1983-01-01','1984-01-01','1985-01-01','1986-01-01','1987-01-01','1988-01-01','1989-01-01',
    '1990-01-01','1991-01-01','1992-01-01','1993-01-01','1994-01-01','1995-01-01','1996-01-01','1997-01-01','1998-01-01','1999-01-01','2000-01-01',
    '2001-01-01','2002-01-01','2003-01-01','2004-01-01','2005-01-01','2006-01-01','2007-01-01','2008-01-01','2009-01-01','2010-01-01','2011-01-01',
    '2012-01-01','2013-01-01','2014-01-01','2015-01-01','2016-01-01','2017-01-01'
    ]

    rea         = xr.open_dataset(sys.argv[1])
    future      = sys.argv[2]
    rg_lat      = -31.53
    rg_long     = -49.86
    rg_rea      = get_data(rea, rg_lat, rg_long)
    path        = './mapes_historic/'
    Parallel(n_jobs=-1,backend='multiprocessing')(delayed(predict_historic)(rg_rea, inicio, path, future) for inicio in inicios_datasets)


    print('###############################################')
    print('###############################################')
    print('##                                           ##')
    print('##      Simulation succesfully finished!     ##')
    print('##                                           ##')
    print('###############################################')
    print('###############################################')