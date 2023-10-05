#!/bin/bash
### Job Name
#PBS -N job_arrays_jhayron
### Project code
#PBS -A UMCP0021
#PBS -l walltime=03:00:00
#PBS -q share
### Merge output and error files
#PBS -j oe
#PBS -k eod
### Select one CPU
#PBS -l select=1:ncpus=1:mem=100GB
### Specify index range of sub-jobs
#PBS -J 1-3

# Define variables and origins lists

# variables=('MLD' 'OHC100' 'OHC200' 'OHC300' 'OHC50' 'OHC700' 'SSH' 'SST'
#             'OLR' 'SD' 'STL_1m' 'STL_full' 'SWVL_1m' 'SWVL_full' 'U10' 'U200' 'Z500'
#             'IC' 'IT' 'SST')
# origins=('SODA' 'SODA' 'SODA' 'SODA' 'SODA' 'SODA' 'SODA' 'SODA'
#           'ERA5' 'ERA5' 'ERA5' 'ERA5' 'ERA5' 'ERA5' 'ERA5' 'ERA5' 'ERA5'
#           'SODA' 'IT' 'SST')
          
variables=('SSH' 'SST' 'OLR')
origins=('SODA' 'SODA' 'ERA5')

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

# Function to execute subjob for index PBS_ARRAY_INDEX and log outputs
execute_subjob() {
    module load conda
    conda activate weather_regimes
    python /glade/u/home/jhayron/WR_Predictability/1_DataPreparation/JoinDatasets.py ${variables[$PBS_ARRAY_INDEX - 1]} ${origins[$PBS_ARRAY_INDEX - 1]} > output.$PBS_ARRAY_INDEX 2>&1
}

# Loop through the indices in the array
for index in $(seq $PBS_ARRAY_INDEX $((PBS_ARRAY_INDEX + 3)))
do
    # Execute subjob for the current index and log outputs
    execute_subjob $index
done
