import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import datetime as dt
import copy

print('hola')
print(sys.argv)

def get_full_dataset(path_files, var, origin):
    if origin == 'ERA5':
        path_var = f'{path_files}{origin}/Daily_05Deg/{var}/'
    elif (origin == 'SODA'):
        path_var = f'{path_files}{origin}_Daily/{var}/'
    elif (origin == 'OISSTv2'):
        path_var = f'{path_files}{origin}/{var}/'

    list_files = np.sort(glob.glob(f'{path_var}*.nc'))#[4000:5000]
    dataset = xr.open_mfdataset(list_files,combine='nested',concat_dim="time")
    if dataset.time.dtype==np.int64:
        dataset['time'] = np.array([dt.datetime.strptime(list_files[i].split('_')[-1],'%Y-%m-%d.nc')\
            for i in range(len(list_files))])
    dataset = dataset.where(dataset.time>np.datetime64('1981-01-01'),drop=True)
    dataset = dataset.load()
    if origin == 'SODA':
        if 'xt_ocean' in dataset.dims:
            dataset = dataset.rename({'xt_ocean': 'lon','yt_ocean': 'lat'})
        elif 'xt' in dataset.dims:
            dataset = dataset.rename({'xt': 'lon','yt': 'lat'})
    if origin == 'OISSTv2':
        dataset = dataset.assign_coords(lat=dataset.lat.values[:,0],lon=dataset.lon.values[0,:])
        dataset = dataset.rename({'x': 'lon', 'y': 'lat'})
    if origin == 'ERA5':
        dataset = dataset.assign_coords(lat=dataset.lat.values[:,0],lon=dataset.lon.values[0,:])
        dataset = dataset.rename({'x': 'lon', 'y': 'lat'})
    return dataset

# Path to files and other constants
path_files = '/glade/work/jhayron/Data4Predictability/'
path_daily_datasets = '/glade/scratch/jhayron/Data4Predictability/DailyDatasets/'

# Get variable and origin from command-line arguments
variable = sys.argv[1]
origin = sys.argv[2]

print(f'Processing variable: {variable}, origin: {origin}')
dataset = get_full_dataset(path_files, variable, origin)
var_name_xarray = list(dataset.data_vars.keys())[0]
dataset.to_netcdf(f'{path_daily_datasets}{variable}_{origin}.nc')
print(f'Finished succesfully')
