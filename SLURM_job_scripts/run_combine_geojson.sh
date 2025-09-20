#!/bin/bash
#SBATCH --job-name=combine_geojson
#SBATCH --output=log/combine_geojson_%j.out
#SBATCH --error=log/combine_geojson_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
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

# Create log directory
mkdir -p log

echo "Starting GeoJSON combination process..."
echo "Using 1 CPU core with 64GB memory"
echo "Processing files in batches of 20 to avoid memory issues"

# Run GeoJSON combination script
echo "=========================================="
echo "Running combine_geojson_folder.py"
echo "=========================================="
echo "Start time: $(date)"
$PYTHON_PATH scripts/utilities/combine_geojson_folder.py
EXIT_CODE=$?
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "GeoJSON combination completed successfully"
else
    echo "GeoJSON combination failed with exit code $EXIT_CODE"
    exit 1
fi

echo ""
echo "GeoJSON combination completed successfully!"
echo "Job finished at:" $(date)
