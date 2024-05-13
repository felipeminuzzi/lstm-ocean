import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import sys
#import tensorflow as tf
import csv
import time
import warnings
import multiprocessing as mp
from sklearn import preprocessing
warnings.filterwarnings('ignore')

def get_data(dataset, lat, lon):
    rg_lat      = lat
    rg_long     = lon
    latlon      = dataset.sel(latitude=rg_lat, longitude=rg_long, method=None)
    rg_rea      = pd.DataFrame({
        'time'      : pd.to_datetime(dataset.time.values),
        'Hs'        : latlon.swh.values,
        '10m-direc' : latlon.dwi.values,
        '10m-speed' : latlon.wind.values,
        'Period'    : latlon.pp1d.values
    })
    rg_rea      = rg_rea.set_index('time')
    return rg_rea

def create_train_dataset(x, df, add_step):
    """
    This bla bla bla
    """    
    positions         = {}
    positions['east']       = [x[0]+add_step, x[1]]
    positions['west']       = [x[0]-add_step, x[1]]
    positions['north']      = [x[0], x[1]+add_step]
    positions['south']      = [x[0], x[1]-add_step]
    positions['south-east'] = [x[0]+add_step, x[1]-add_step]
    positions['south-west'] = [x[0]-add_step, x[1]-add_step]
    positions['north-east'] = [x[0]+add_step, x[1]+add_step]
    positions['north-west'] = [x[0]-add_step, x[1]+add_step]
    
    rg_rea                  = get_data(df, x[1], x[0])
    for key in positions.keys():

        try:        
            aux_df              = get_data(df, positions[key][1], positions[key][0])
            aux_df.rename(columns = {'Hs': f'{key}_Hs', 
                                    '10m-direc': f'{key}_10m-direc', 
                                    '10m-speed': f'{key}_10m-speed', 
                                    'Period': f'{key}_Period'},
                                    inplace=True)
            rg_rea              = pd.concat([rg_rea, aux_df], axis=1)
        except:
            continue
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
            seq_x, seq_y = sequence[i:end_ix, 1:], sequence[end_ix+lead_time-1, 0]
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
        epochs               = 1,
        validation_split     = 0.2,
        callbacks            = [early_stopping],
        verbose              = 0
    )
    return history

def prediction(model, data_in, n_steps_in, num_features):
    x_predict                    = data_in.reshape(1, n_steps_in, num_features)
    predict                      = model.predict(x_predict)
    return predict[0,0]

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

def get_model(num_prev, num_features):
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

    inputs_1                 = df['Hs'][:-(npredict+count)].values
    inputs_2                 = df['Period'][:-(npredict+count)].values
    inputs_3                 = df['10m-direc'][:-(npredict+count)].values
    inputs_4                 = df['10m-speed'][:-(npredict+count)].values

    inputs_1                 = inputs_1.reshape((len(inputs_1), 1))
    inputs_2                 = inputs_2.reshape((len(inputs_2), 1))
    inputs_3                 = inputs_3.reshape((len(inputs_3), 1))
    inputs_4                 = inputs_4.reshape((len(inputs_4), 1))

    inputs                   = np.hstack((inputs_1, inputs_2, inputs_3, inputs_4))
    X, y                     = split_sequence(inputs, forecast, lead, False)

    model                    = get_model(1, num_features)
    history                  = compile_and_fit(model, X, y)

    x_input_1                = df['Hs'][-(npredict+count):].values
    x_input_2                = df['Period'][-(npredict+count):].values
    x_input_3                = df['10m-direc'][-(npredict+count):].values
    x_input_4                = df['10m-speed'][-(npredict+count):].values

    x_input_1                = x_input_1.reshape((len(x_input_1), 1))
    x_input_2                = x_input_2.reshape((len(x_input_2), 1))
    x_input_3                = x_input_3.reshape((len(x_input_3), 1))
    x_input_4                = x_input_4.reshape((len(x_input_4), 1))

    x_input                  = np.hstack((x_input_1, x_input_2, x_input_3, x_input_4))
    x_in, labels             = split_sequence(x_input, forecast, lead, False)
    
    result                   = pd.DataFrame()
    result['Data']           = df.index[-npredict+forecast+5:]
    result['Hs Reanalysis Value'] = labels
    
    predictions              = []
    predictions              = [prediction(model, dado, forecast, num_features) for dado in x_in]
    result['Hs Predict Value'] = predictions
    
    mape_model           = mape(result['Hs Reanalysis Value'], result['Hs Predict Value'])
    error                = erro(result['Hs Reanalysis Value'], result['Hs Predict Value'])
    
    print(f'MAPE for lead time {lead}: {mape_model}')
    result.to_csv(f'lstm_predictions_{lead}_leadtime.csv')
    plot(result,error,lead,mape_model) 

        
path         = '/Users/felipeminuzzi/Documents/OCEANO/Simulations/Machine_Learning/era5_reanalysis_utlimos_dados.nc'
data_era     = xr.open_dataset(path)
lats         = data_era.latitude.values
longs        = data_era.longitude.values
ni           = len(longs)
nj           = len(lats)
# rg_lat      = -31.53
# rg_long     = -49.86
add_step     = 0.5     

yv, xv       = np.meshgrid(lats, longs)
df_latlong   = pd.DataFrame(dict(long=xv.ravel(), lat=yv.ravel()))
lst_latlong  = df_latlong.values

for x in lst_latlong:
    df_train = create_train_dataset(x,data_era, add_step)
    breakpoint()

column_ind   = {name: i for i, name in enumerate(rg_rea.columns)}

lead_time    = [6,12,18,24]
forecast     = 12
num_features = 3
npredict     = 744

start                        = time.time()
for lead in lead_time:
    create_future_predict(lead, rg_rea, npredict, forecast, num_features)

end                          = time.time()

print('Time of execution: ',(end-start)/60, ' minutes.')
print('###############################################')
print('###############################################')
print('##                                           ##')
print('##      Simulation succesfully finished!     ##')
print('##                                           ##')
print('###############################################')
print('###############################################')