#!/bin/bash
### Job Name
#PBS -N hyper_parameters_week3
### Project code
#PBS -A UMCP0021
#PBS -l walltime=02:00:00
#PBS -q casper
### Merge output and error files
#PBS -j oe
#PBS -k eod
### Select one CPU
#PBS -l select=1:ncpus=1:mem=2GB
### Specify index range of sub-jobs
#PBS -J 1-9

variables=('Z500_ERA5' 'U10_ERA5' 'OLR_ERA5' 'SWVL_full_ERA5' 'SD_ERA5' 'STL_full_ERA5' 'IT_SODA' 'OHC100_SODA' 'SST_SODA')

echo "Processing variable: ${variables[$PBS_ARRAY_INDEX - 1]}"
module load conda
conda activate cnn_wr
python /glade/u/home/jhayron/WR_Predictability/3_MLModels/OptunaOptimization.py ${variables[$PBS_ARRAY_INDEX - 1]} 3 > output.$PBS_ARRAY_INDEX 2>&1
echo "Finished processing variable: ${variables[$PBS_ARRAY_INDEX - 1]}"
