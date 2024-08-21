import xarray as xr 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import csv
import sys

fn1  = xr.open_dataset(sys.argv[1])

lat     = fn1.latitude.values
lon     = fn1.longitude.values
swh     = fn1.swh.values
# dwi     = fn1.dwi.values
# wind    = fn1.wind.values
# pp1d    = fn1.pp1d.values
#print(fn1.variables)
# print(fn3.var)
# junto = xr.concat((fn1,fn2,fn3), dim='time')
# print(junto)
# junto.to_netcdf('era20_ensemble_2010_2018.nc')

def to_csv(var_name):
    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';', lineterminator='\n')
        for i in range(len(lat)):
            writer.writerow(var_name[temp][i][:])

def graph(var_name,temp):
    nanvalues = np.ma.masked_where(var_name[temp] == np.nan, var_name[temp])
    fig, ax = plt.subplots()
    llons, llats = np.meshgrid(lon, lat)
    hs_plot = ax.contourf(llons, llats, var_name[temp], 120, cmap='jet')
    plt.colorbar(hs_plot)
    ax.set_facecolor('darkgray') 
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

def to_csv_ensemble(var_name,member):
    with open('data_ensemble.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';', lineterminator='\n')
        for i in range(len(lat)//2):
            writer.writerow(var_name[temp][member][i][:])

def graph_ensemble(var_name,member):
    fig, ax = plt.subplots()
    llons, llats = np.meshgrid(lon, lat)
    hs_plot = ax.contourf(llons, llats, var_name[temp][member], 60, cmap='jet')
    plt.colorbar(hs_plot)
    ax.set_facecolor('black')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

def get_data(dataset, lat, lon):
    rg_lat      = lat
    rg_long     = lon
    latlon      = dataset.sel(latitude=rg_lat, longitude=rg_long, method='nearest')
    rg_rea      = pd.DataFrame({
        'time'      : pd.to_datetime(dataset.time.values),
        'Hs'        : latlon.swh.values,
        # '10m-direc' : latlon.dwi.values,
        # '10m-speed' : latlon.wind.values,
        # 'Period'    : latlon.pp1d.values
     })
    rg_rea      = rg_rea.sort_values(by='time', ignore_index=True)
    rg_rea      = rg_rea.set_index('time')
    return rg_rea

# #lat/long do ondografo de rio grande: 
# #riogrande   = junto.sel(latitude=-31.53, longitude=-49.86, method='nearest')
# #rg_rea      = junto.sel(latitude=-31.53, longitude=-49.86, method='nearest')
# #fore_temp_series = pd.DataFrame({'time': pd.to_datetime(fn.time.values), 'Hs': riogrande.swh.values})
# #fore_temp_series = fore_temp_series.set_index('time')
# #rea_temp_series = pd.DataFrame({'time': pd.to_datetime(junto.time.values), 'Hs': rg_rea.swh.values})
# #rea_temp_series = rea_temp_series.set_index('time')

# #print(fore_temp_series)
# #print(rea_temp_series)
# '''
# fig, ax = plt.subplots()
# ax.plot(fore_temp_series.index, fore_temp_series['Hs'],rea_temp_series.index, rea_temp_series['Hs'],'--' )
# ax.set_xlabel('Time')
# ax.set_ylabel('Hs (m)')
# ax.legend(['Forecast 06:00','Reanalysis'])
# plt.show()
# '''
# temp   = 0
# graph(wind)
# #to_csv_ensemble(swh,0)
    
graph(swh,0)
plt.show()



# rea = fn
# rg_lat      = -27.24
# rg_long     = -47.15
# # member = 8
# rg_rea      = get_data(fn1, rg_lat, rg_long)
# print(rg_rea)

#rg_rea.to_csv('../Machine_Learning/results/predict_with_historic_buoy_without_outliers/reanalise_to_compare_itajai.csv')

# colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'tomato']

# plt.figure(1)
# for i in range(10):
#     rg_rea      = get_data(r, rg_lat, rg_long, i)
#     create_plots(i, rg_rea)
# rg_rea['Hs'].plot()
# plt.show()