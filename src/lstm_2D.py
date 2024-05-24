import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import tensorflow as tf
import time
from joblib import Parallel, delayed
from tqdm import tqdm

import warnings
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

def format_path(path):
    """""Formats the path string in order to avoid conflicts."""

    if path[-1]!='/':
        path = path + '/'

    if not os.path.exists(path):
        os.makedirs(path)

    return path


def split_sequence(sequence, sequence2, n_steps_in, lead_time):
    X, y = [], []
    m = len(sequence) - lead_time
    for i in range(m):
        end_ix     = i + n_steps_in

        if end_ix > m:
            break
        seq_x, seq_y = sequence[i:end_ix, 0:], sequence2[end_ix+lead_time-1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def prepare_data(X, y, num_features):
    dim_1                        = X.shape[0]
    dim_2                        = X.shape[1]
    dim_y                        = y.shape[0]

    X                            = X.flatten()
    y                            = y.flatten()
    X                            = X.reshape((dim_1, dim_2, num_features))
    y                            = y.reshape((dim_y, 1, 1))
    
    return X,y 

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
        epochs               = 50,
        validation_split     = 0.2,
        callbacks            = [early_stopping],
        verbose              = 1
    )
    return history

def prediction(model, data_in, n_steps_in, num_features):
    x_predict                    = data_in.reshape(1, n_steps_in, num_features)
    predict                      = model.predict(x_predict)
    return predict[0,0]

def plot(df,error,lead, mape_value, path, id):
    plt.figure(1)
    plt.plot(df['Data'], df['Hs Reanalysis Value'], 'r-', label='Label (dado real)')
    plt.plot(df['Data'], df['Hs Predict Value'], 'b--', label='Previsão')
    plt.xlabel('Data')
    plt.ylabel('Wave height (Hs)')
    plt.legend()
    plt.title(f'Previsão de altura de onda para lead time {lead}. MAPE: {mape_value}')
    plt.savefig(path + f'lstm_predicts_lat{id[1]}_long{id[0]}_lead{lead}.png', bbox_inches='tight')

    plt.figure(2)
    plt.plot(df['Data'], error, 'r-')
    plt.xlabel('Data')
    plt.ylabel('Absolute error')
    plt.title(f'Erro absoluto para lead time {lead}. MAPE: {mape_value}')
    plt.savefig(path + f'lstm_erro_lat{id[1]}_long{id[0]}_lead{lead}.png', bbox_inches='tight')

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
        tf.keras.layers.LSTM(256,  activation='relu', return_sequences=True, input_shape=(forecast, num_features)),
        tf.keras.layers.LSTM(128, activation='relu', return_sequences=True, input_shape=(forecast, num_features)),
        tf.keras.layers.LSTM(64,  activation='relu', return_sequences=True, input_shape=(forecast, num_features)),
        tf.keras.layers.LSTM(48,  activation='relu', return_sequences=True, input_shape=(forecast, num_features)),
        tf.keras.layers.LSTM(32,  activation='relu', return_sequences=False),
        tf.keras.layers.Dense(num_prev, kernel_initializer=tf.initializers.zeros)])
    return modelo

def create_train_test(df, npredict, lead, tgt = 'Hs'):
    if lead == 0:
        count = 0
    elif lead==6:
        count = 0
    elif lead==12:
        count = 6
    elif lead==18:
        count = 12
    else:
        count = 18
    
    target                   = df[tgt][:-(npredict+count)]
    target_predict           = df[tgt][-(npredict+count):]

    df.drop([tgt], axis=1, inplace=True)
    datas                    = []

    for col in df.columns:
        inputs_1             = df[col][:-(npredict+count)].values
        inputs_1             = inputs_1.reshape((len(inputs_1), 1))
        datas.append(inputs_1)
    
    inputs                   = np.hstack(datas)

    predict_data = []
    for col in df.columns:
        inputs_2             = df[col][-(npredict+count):].values
        inputs_2             = inputs_2.reshape((len(inputs_2), 1))
        predict_data.append(inputs_2)

    x_input                  = np.hstack(predict_data)
    
    return inputs, target, x_input, target_predict

def future_predict(lead, df, npredict, forecast, num_features, path, id):

    train_input, train_target, test_input, test_target    = create_train_test(df, npredict, lead)
    x_train, y_train                                      = split_sequence(train_input, train_target, forecast, lead)
    X, y                                                  = prepare_data(x_train, y_train, num_features)

    x_test, y_test                                        = split_sequence(test_input, test_target, forecast, lead)
    x_in, labels                                          = prepare_data(x_test, y_test, num_features)
    
    model                    = get_model(1, forecast, num_features)
    history                  = compile_and_fit(model, X, y)
    
    result                   = pd.DataFrame()
    result['Data']           = test_target.index[-len(labels):]
    result['Hs Reanalysis Value'] = y_test
    
    predictions              = []
    predictions              = [prediction(model, dado, forecast, num_features) for dado in x_in]
    result['Hs Predict Value'] = predictions
    
    mape_model           = mape(result['Hs Reanalysis Value'], result['Hs Predict Value'])
    error                = erro(result['Hs Reanalysis Value'], result['Hs Predict Value'])
    
    print(f'MAPE for lead time {lead}: {mape_model}')
    result.to_csv(path + f'lstm_predictions_lat{id[1]}_long{id[0]}_lead{lead}.csv')
    plot(result,error,lead,mape_model, path, id) 

def dispatch(x, data_era, add_step, lead_time, forecast, npredict, path):

    for lead in lead_time:
        df_train     = create_train_dataset(x, data_era, add_step)
        df_train     = df_train.loc[df_train.index >= pd.to_datetime('2017-02-01')]
        
        num_features = df_train.shape[1] - 1        

        future_predict(lead, df_train, npredict, forecast, num_features, path, x)
        plt.close('all')
        
root_path    = os.getcwd()             
path         = root_path + '/era5_reanalysis_utlimos_dados.nc'
save_path    = format_path(root_path + '/2D_results/')

data_era     = xr.open_dataset(path)
lats         = data_era.latitude.values
longs        = data_era.longitude.values
ni           = len(longs)
nj           = len(lats)
add_step     = 0.5     
lead_time    = [0]
forecast     = 12
npredict     = 744

yv, xv       = np.meshgrid(lats, longs)
df_latlong   = pd.DataFrame(dict(long=xv.ravel(), lat=yv.ravel()))
lst_latlong  = df_latlong.values

start                        = time.time()
Parallel(n_jobs=1,backend='multiprocessing')(delayed(dispatch)(x, data_era, add_step, lead_time, 
                                                                forecast, npredict, save_path) for x in tqdm(lst_latlong, desc='2D prediction...'))    
end                          = time.time()


print('Time of execution: ',(end-start)/60, ' minutes.')
print('###############################################')
print('###############################################')
print('##                                           ##')
print('##      Simulation succesfully finished!     ##')
print('##                                           ##')
print('###############################################')
print('###############################################')