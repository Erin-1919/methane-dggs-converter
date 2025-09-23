#!/bin/bash
#SBATCH --job-name=combine_edgar_csv
#SBATCH --output=log/combine_edgar_csv_%j.out
#SBATCH --error=log/combine_edgar_csv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=100G
#SBATCH --partition=cpu2019
#SBATCH --mail-user=mingke.li@ucalgary.ca
#SBATCH --mail-type=END,FAIL

# Script directory for the project on HPC
SCRIPTDIR=/home/mingke.li/methane_grid_calculation_ARC
cd $SCRIPTDIR || { echo "Directory $SCRIPTDIR not found"; exit 1; }

echo "Job starting at:" $(date)

# Load conda environment
export PATH=/home/mingke.li/miniconda3/bin:$PATH
source /home/mingke.li/miniconda3/etc/profile.d/conda.sh
conda activate netcdf_dggs_converter

# Python path and environment variables
export PYTHON_PATH="/home/mingke.li/miniconda3/envs/netcdf_dggs_converter/bin/python"
export OMP_NUM_THREADS=1  # Prevent OpenMP from using all cores

# Create log directory
mkdir -p log

echo "Starting EDGAR CSV combination process..."
echo "Using 1 CPU core with 64GB memory"

# Optional parameters via environment variables
# START_YEAR and END_YEAR default to full EDGAR range
START_YEAR=${START_YEAR:-1970}
END_YEAR=${END_YEAR:-2022}
TEST_CSV_FOLDER=${TEST_CSV_FOLDER:-test/test_EDGAR_csv}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-output}

echo "=========================================="
echo "Running combine_edgar_intermediate_to_final.py"
echo "=========================================="
echo "Start time: $(date)"
echo "Parameters: --start_year $START_YEAR --end_year $END_YEAR"
echo "            --test_csv_folder $TEST_CSV_FOLDER"
echo "            --output_folder $OUTPUT_FOLDER"

$PYTHON_PATH scripts/utilities/combine_edgar_intermediate_to_final.py \
  --start_year $START_YEAR \
  --end_year $END_YEAR \
  --test_csv_folder $TEST_CSV_FOLDER \
  --output_folder $OUTPUT_FOLDER

EXIT_CODE=$?
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "EDGAR CSV combination completed successfully"
else
    echo "EDGAR CSV combination failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "EDGAR CSV combination job finished at:" $(date)


