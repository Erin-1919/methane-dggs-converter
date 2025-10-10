#!/bin/bash
#SBATCH --job-name=create_coverage_geoms
#SBATCH --output=log/create_coverage_geoms_%j.out
#SBATCH --error=log/create_coverage_geoms_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --partition=cpu2023
#SBATCH --mail-user=mingke.li@ucalgary.ca
#SBATCH --mail-type=END,FAIL

# Set script directory (adjust if repository is elsewhere)
SCRIPTDIR=/home/mingke.li/methane_grid_calculation_ARC
cd "$SCRIPTDIR" || { echo "Directory $SCRIPTDIR not found"; exit 1; }

echo "Job starting at: $(date)"

# Load conda environment
export PATH=/home/mingke.li/miniconda3/bin:$PATH
source /home/mingke.li/miniconda3/etc/profile.d/conda.sh
conda activate netcdf_dggs_converter

# Set Python path and environment variables
export PYTHON_PATH="/home/mingke.li/miniconda3/envs/netcdf_dggs_converter/bin/python"

# Derive CPU counts from SLURM env; fallback to 1
if [[ -n "${SLURM_CPUS_PER_TASK}" && "${SLURM_CPUS_PER_TASK}" -gt 0 ]]; then
  export NUM_CORES=${SLURM_CPUS_PER_TASK}
elif [[ -n "${SLURM_NTASKS}" && -n "${SLURM_CPUS_PER_TASK}" ]]; then
  export NUM_CORES=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
else
  export NUM_CORES=1
fi

# Avoid over-subscription by libraries using OpenMP/BLAS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Create log directory
mkdir -p log

echo "Creating coverage geometries with ${NUM_CORES} CPU cores"
echo "Start time: $(date)"

# Run the coverage geometry creation script
$PYTHON_PATH scripts/analysis/create_dggs_coverage_geojsons.py
EXIT_CODE=$?

echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
  echo "Coverage geometry creation completed successfully"
else
  echo "Coverage geometry creation failed with exit code $EXIT_CODE"
  exit $EXIT_CODE
fi

echo "Job finished at: $(date)"