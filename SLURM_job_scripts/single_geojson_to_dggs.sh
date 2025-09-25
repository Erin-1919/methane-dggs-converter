#!/bin/bash
#SBATCH --job-name=dggs_single_region
#SBATCH --output=log/dggs_single_region.out
#SBATCH --error=log/dggs_single_region.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:0
#SBATCH --mem=80G
#SBATCH --partition=cpu2019
#SBATCH --mail-user=mingke.li@ucalgary.ca
#SBATCH --mail-type=END,FAIL

# Set script directory (match your repo path on the cluster)
SCRIPTDIR=/home/mingke.li/methane_grid_calculation_ARC
cd $SCRIPTDIR || { echo "Directory $SCRIPTDIR not found"; exit 1; }

echo "Job starting at:" $(date)

# Load conda environment
export PATH=/home/mingke.li/miniconda3/bin:$PATH
source /home/mingke.li/miniconda3/etc/profile.d/conda.sh
conda activate netcdf_dggs_converter

# Set Python path and environment variables
export PYTHON_PATH="/home/mingke.li/miniconda3/envs/netcdf_dggs_converter/bin/python"
export OMP_NUM_THREADS=1
export NUM_CORES=1

# Create log directory
mkdir -p log

echo "Running single-region DGGS conversion"
echo "Start time: $(date)"

# Parameters (override via environment or edit here)
: ${GRID_TYPE:=rhealpix}
: ${LEVEL:=10}
: ${INPUT_PATH:=data/geojson/newyorkstate.geojson}
: ${OUTPUT_DIR:=data/geojson/regional_grid}
: ${OUT_FORMAT:=parquet}   # parquet | geojsonl | geojsonl.gz | geojson | geojson.gz
: ${TILE_DEG:=2.0}
: ${BATCH_SIZE:=10000}
: ${DEDUP:=1}              # 1 enables --dedup

CMD_ARGS=(
  --grid "$GRID_TYPE"
  --level "$LEVEL"
  --input "$INPUT_PATH"
  --output-dir "$OUTPUT_DIR"
  --format "$OUT_FORMAT"
  --tile-deg "$TILE_DEG"
  --batch-size "$BATCH_SIZE"
)

if [ "$DEDUP" = "1" ]; then
  CMD_ARGS+=(--dedup)
fi

echo "Command args: ${CMD_ARGS[*]}"
$PYTHON_PATH scripts/dggs_grid_creation/convert_single_geojson_to_dggs.py ${CMD_ARGS[*]}
EXIT_CODE=$?
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Single-region conversion completed successfully"
else
    echo "Single-region conversion failed with exit code $EXIT_CODE"
    exit 1
fi

echo "Job finished at:" $(date)


