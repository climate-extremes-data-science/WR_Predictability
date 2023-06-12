import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import glob
import os
import matplotlib.pyplot as plt
import xesmf as xe
import gc
import sys

#variables = ['OLR', 'SD', 'STL_1m', 'STL_full', 'SWVL_1m', 'SWVL_full', 'U10', 'U200', 'Z500']
#variables_ds = ['MTNLWRF', 'SD', 'STL_1m', 'STL_full', 'SWVL_1m', 'SWVL_full', 'U', 'U', 'Z']
#units = ['W/m^2', 'm', 'K', 'K', 'm^3/m^3', 'm^3/m^3', 'm/s', 'm/s', 'm^2/s^2']

variables = ['STL_28cm', 'STL_7cm', 'SWVL_28cm', 'SWVL_7cm']
variables_ds = ['STL_28cm', 'STL_7cm', 'SWVL_28cm', 'SWVL_7cm']
units = ['K', 'K', 'm^3/m^3', 'm^3/m^3']

path_era5_daily = '/glade/work/jhayron/Data4Predictability/ERA5/Daily/'
path_code = u'/glade/u/home/jhayron/WR_Predictability/1_DataPreparation/regrid.py'
for ivar in range(len(variables)):
    list_files = np.sort(glob.glob(f'{path_era5_daily}{variables[ivar]}/*.nc'))
    for ifile in range(len(list_files)):
        path_temp = list_files[ifile]
        if ifile == 0:
            save_regridder = 'True'
        else:
            save_regridder = 'False'
        # print(f'python {path_code} {path_temp} {variables[ivar]} {variables_ds[ivar]} {units[ivar]} {save_regridder}')
        os.system(f'python {path_code} {path_temp} {variables[ivar]} {variables_ds[ivar]} {units[ivar]} {save_regridder}')