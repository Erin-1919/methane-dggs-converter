#!/bin/bash
#SBATCH --job-name=dggs_conversion
#SBATCH --output=log/dggs_conversion.out
#SBATCH --error=log/dggs_conversion.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:0
#SBATCH --mem=32G
#SBATCH --partition=cpu2019
#SBATCH --mail-user=mingke.li@ucalgary.ca
#SBATCH --mail-type=END,FAIL

# Set script directory
SCRIPTDIR=/home/mingke.li/methane_grid_calculation_ARC
cd $SCRIPTDIR || { echo "Directory $SCRIPTDIR not found"; exit 1; }

echo "Job starting at:" $(date)

# Load conda environment
export PATH=/home/mingke.li/miniconda3/bin:$PATH
source /home/mingke.li/miniconda3/etc/profile.d/conda.sh
conda activate netcdf_dggs_converter

# Set Python path and environment variables
export PYTHON_PATH="/home/mingke.li/miniconda3/envs/netcdf_dggs_converter/bin/python"
export OMP_NUM_THREADS=1  # Prevent OpenMP from using all cores
export NUM_CORES=16       # Match the number of CPUs requested

# Create log directory
mkdir -p log

echo "Starting DGGS conversion process..."
echo "Using $NUM_CORES CPU cores for parallel processing"

# Run country GeoJSON to DGGS conversion
echo "=========================================="
echo "Running country GeoJSON to DGGS conversion"
echo "=========================================="
echo "Start time: $(date)"
$PYTHON_PATH scripts/dggs_grid_creation/convert_country_geojson_to_dggs.py
COUNTRY_EXIT_CODE=$?
echo "End time: $(date)"

if [ $COUNTRY_EXIT_CODE -eq 0 ]; then
    echo "Country conversion completed successfully"
else
    echo "Country conversion failed with exit code $COUNTRY_EXIT_CODE"
    exit 1
fi

echo ""

# Run offshore GeoJSON to DGGS conversion
echo "=========================================="
echo "Running offshore GeoJSON to DGGS conversion"
echo "=========================================="
echo "Start time: $(date)"
$PYTHON_PATH scripts/dggs_grid_creation/convert_offshore_to_dggs.py
OFFSHORE_EXIT_CODE=$?
echo "End time: $(date)"

if [ $OFFSHORE_EXIT_CODE -eq 0 ]; then
    echo "Offshore conversion completed successfully"
else
    echo "Offshore conversion failed with exit code $OFFSHORE_EXIT_CODE"
    exit 1
fi

echo ""
echo "All DGGS conversions completed successfully!"
echo "Job finished at:" $(date)
