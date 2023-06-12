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

# variables = ['OLR', 'SD', 'STL_1m', 'STL_full', 'SWVL_1m', 'SWVL_full', 'U10', 'U200', 'Z500']
# variables_ds = ['MTNLWRF', 'SD', 'STL_1m', 'STL_full', 'SWVL_1m', 'SWVL_full', 'U', 'U', 'Z']
# units = ['W/m^2', 'm', 'K', 'K', 'm^3/m^3', 'm^3/m^3', 'm/s', 'm/s', 'm^2/s^2']

variables = ['SST']
variables_ds = ['sst']
units = ['degC']

path_oisst_daily = '/glade/collections/rda/data/ds277.7/avhrr_v2.1/'

def list_files_recursive(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return np.sort(file_list)

path_code = u'/glade/u/home/jhayron/WR_Predictability/1_DataPreparation/regrid_oisst.py'

for ivar in range(len(variables)):
    list_files = list_files_recursive(path_oisst_daily)
    for ifile in range(len(list_files)):
        path_temp = list_files[ifile]
        if ifile == 0:
            save_regridder = 'True'
        else:
            save_regridder = 'False'
        # print(f'python {path_code} {path_temp} {variables[ivar]} {variables_ds[ivar]} {units[ivar]} {save_regridder}')
        os.system(f'python {path_code} {path_temp} {variables[ivar]} {variables_ds[ivar]} {units[ivar]} {save_regridder}')
