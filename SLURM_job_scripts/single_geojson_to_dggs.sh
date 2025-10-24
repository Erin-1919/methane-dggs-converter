#!/bin/bash
#SBATCH --job-name=dggs_single_region
#SBATCH --output=log/dggs_single_regionm.out
#SBATCH --error=log/dggs_single_regionm.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:0
#SBATCH --mem=80G
#SBATCH --partition=cpu2021
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
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUM_CORES=${SLURM_CPUS_PER_TASK}

# Create log directory
mkdir -p log

echo "Running single-region DGGS conversion"
echo "Start time: $(date)"

# Parameters (override via environment or edit here). Leave empty to use Python defaults.
: ${GRID_TYPE:=}
: ${LEVEL:=}
: ${INPUT_PATH:=}
: ${OUTPUT_DIR:=}
: ${OUT_FORMAT:=}   # parquet | geojsonl | geojsonl.gz | geojson | geojson.gz
: ${TILE_DEG:=}
: ${BATCH_SIZE:=}
: ${DEDUP:=1}              # 1 enables --dedup

CMD_ARGS=()

# Only pass flags that are explicitly set; otherwise rely on Python script defaults
[ -n "$GRID_TYPE" ] && CMD_ARGS+=(--grid "$GRID_TYPE")
[ -n "$LEVEL" ] && CMD_ARGS+=(--level "$LEVEL")
[ -n "$INPUT_PATH" ] && CMD_ARGS+=(--input "$INPUT_PATH")
[ -n "$OUTPUT_DIR" ] && CMD_ARGS+=(--output-dir "$OUTPUT_DIR")
[ -n "$OUT_FORMAT" ] && CMD_ARGS+=(--format "$OUT_FORMAT")
[ -n "$TILE_DEG" ] && CMD_ARGS+=(--tile-deg "$TILE_DEG")
[ -n "$BATCH_SIZE" ] && CMD_ARGS+=(--batch-size "$BATCH_SIZE")

# Always enable parallelism and low-memory dataset mode
CMD_ARGS+=(--workers "$SLURM_CPUS_PER_TASK")
CMD_ARGS+=(--parquet-dataset)

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


