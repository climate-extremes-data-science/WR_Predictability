#!/bin/bash
### Job Name
#PBS -N job_arrays_jhayron
### Project code
#PBS -A UMCP0021
#PBS -l walltime=12:00:00
#PBS -q economy
### Merge output and error files
#PBS -j oe
#PBS -k eod
### Select one CPU
#PBS -l select=1:ncpus=1:mem=109GB
### Specify index range of sub-jobs
#PBS -J 1-2

# Define variables and origins lists

variables=('OHC700')
origins=('ERA5')

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

# Function to execute subjob for index PBS_ARRAY_INDEX and log outputs
execute_subjob() {
    module load conda
    conda activate weather_regimes
    python /glade/u/home/jhayron/WR_Predictability/1_DataPreparation/JoinDatasets_Parts.py ${variables[$PBS_ARRAY_INDEX - 1]} ${origins[$PBS_ARRAY_INDEX - 1]} > output.$PBS_ARRAY_INDEX 2>&1
}

# Loop through the indices in the array
for index in $(seq $PBS_ARRAY_INDEX $((PBS_ARRAY_INDEX + 2)))
do
    # Get the current variable and origin
    variable=${variables[$index - 1]}
    origin=${origins[$index - 1]}

    # Print the variable and origin that started to execute
    echo "Processing variable: $variable, origin: $origin"

    # Execute subjob for the current index and log outputs
    execute_subjob $index

    echo "Finished processing variable: $variable, origin: $origin"
done
