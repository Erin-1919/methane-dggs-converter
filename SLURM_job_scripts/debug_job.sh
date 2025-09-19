#!/bin/bash
#SBATCH --job-name=test_conda
#SBATCH --output=log/test_conda.out
#SBATCH --error=log/test_conda.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:0
#SBATCH --mem=2G
#SBATCH --partition=single
#SBATCH --mail-user=mingke.li@ucalgary.ca
#SBATCH --mail-type=END,FAIL

# Set script directory
SCRIPTDIR=/home/mingke.li/methane_grid_calculation_ARC
cd $SCRIPTDIR || { echo "Directory $SCRIPTDIR not found"; exit 1; }

echo "Job starting at:" $(date)

# Create log directory if it doesn't exist
mkdir -p log

# Load conda
export PATH=/home/mingke.li/miniconda3/bin:$PATH
source /home/mingke.li/miniconda3/etc/profile.d/conda.sh
conda activate netcdf_dggs_converter

# Set the correct Python path for the environment
export PYTHON_PATH="/home/mingke.li/miniconda3/envs/netcdf_dggs_converter/bin/python"

echo "=== ENVIRONMENT DEBUG INFO ==="
echo "Current user: $(whoami)"
echo "Current directory: $(pwd)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo ""

echo "=== CONDA ENVIRONMENT INFO ==="
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "CONDA_EXE: $CONDA_EXE"
echo "CONDA_PYTHON_EXE: $CONDA_PYTHON_EXE"
echo ""

echo "=== PYTHON INFO ==="
echo "Python path: $PYTHON_PATH"
echo "Python version: $($PYTHON_PATH --version)"
echo "Python executable exists: $([ -f "$PYTHON_PATH" ] && echo "YES" || echo "NO")"
echo ""

echo "=== PATH INFO ==="
echo "PATH: $PATH"
echo ""

echo "=== PACKAGE TESTING ==="
echo "Testing required packages:"
packages=("geopandas" "pandas" "numpy" "shapely" "fiona" "pyproj" "json" "subprocess" "os" "typing" "multiprocessing")
for package in "${packages[@]}"; do
    if $PYTHON_PATH -c "import $package; print('✓ $package:', getattr(__import__('$package'), '__version__', 'unknown'))" 2>/dev/null; then
        echo "✓ $package: Available"
    else
        echo "✗ $package: NOT FOUND"
    fi
done
echo ""

echo "=== CONDA LIST OUTPUT ==="
$PYTHON_PATH -m pip list | grep -E "(geopandas|pandas|numpy|shapely|fiona|pyproj)" || echo "No matching packages found in pip list"
echo ""

echo "=== DIRECTORY STRUCTURE ==="
echo "Project directory contents:"
ls -la
echo ""
echo "Scripts directory contents:"
ls -la scripts/dggs_grid_creation/ || echo "Scripts directory not found"
echo ""

echo "=== TESTING DGGS COMMAND ==="
which dgg || echo "dgg command not found in PATH"
echo ""

echo "=== FINAL PYTHON TEST ==="
$PYTHON_PATH -c "
import sys
print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('Python path:', sys.path[:3])
try:
    import geopandas as gpd
    print('✓ geopandas imported successfully, version:', gpd.__version__)
except ImportError as e:
    print('✗ geopandas import failed:', e)
"

echo ""
echo "Job finished at:" $(date)
