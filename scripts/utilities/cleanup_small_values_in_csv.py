import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np


def setup_logger() -> logging.Logger:
    os.makedirs("log", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("log", f"cleanup_small_values_in_csv_{ts}.log")
    logger = logging.getLogger(f"cleanup_small_values_{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    logger.info(f"Logging initialized: {log_path}")
    return logger


def discover_csv_files(path: str) -> List[str]:
    csv_files = []
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
    elif os.path.isfile(path) and path.lower().endswith('.csv'):
        csv_files.append(path)
    return csv_files


def find_id_columns(columns: List[str], user_ids: List[str]) -> List[str]:
    lower_to_original = {c.lower(): c for c in columns}
    present: List[str] = []
    for uid in user_ids:
        if uid.lower() in lower_to_original:
            present.append(lower_to_original[uid.lower()])
    return present


def clean_file_chunked(csv_path: str, threshold: float, id_candidates: List[str], inplace: bool, 
                      chunk_size: int, logger: logging.Logger) -> Tuple[str, int, int, int]:
    logger.info(f"Processing file: {csv_path}")
    
    # First pass: identify columns and get file info
    try:
        # Read just the header to identify columns
        sample_df = pd.read_csv(csv_path, nrows=0)
        columns = list(sample_df.columns)
    except Exception as e:
        logger.error(f"  Failed to read CSV header: {e}")
        return csv_path, 0, 0, 0

    if not columns:
        logger.info("  File is empty; skipping")
        return csv_path, 0, 0, 0

    # Identify identifier columns (case-insensitive): default candidates dggsID, GID, Year/year
    id_cols_present = find_id_columns(columns, id_candidates)
    if not id_cols_present:
        logger.warning("  No identifier columns found; will treat all columns as variables")
    var_cols = [c for c in columns if c not in id_cols_present]
    if not var_cols:
        logger.info("  No variable columns to process; skipping")
        return csv_path, 0, 0, 0

    # Output path
    if inplace:
        out_path = csv_path + ".tmp"
    else:
        base, ext = os.path.splitext(csv_path)
        out_path = f"{base}_cleaned{ext}"

    # Process file in chunks
    total_small_count = 0
    total_rows_before = 0
    total_rows_after = 0
    first_chunk = True
    
    try:
        for chunk_df in pd.read_csv(csv_path, chunksize=chunk_size):
            if chunk_df.empty:
                continue
                
            total_rows_before += len(chunk_df)
            
            # Ensure numeric for variable columns
            for c in var_cols:
                if not np.issubdtype(chunk_df[c].dtype, np.number):
                    chunk_df[c] = pd.to_numeric(chunk_df[c], errors='coerce')

            # Count and zero-out values with absolute magnitude < threshold
            values_before = chunk_df[var_cols].to_numpy(copy=False)
            mask_small = np.isfinite(values_before) & (np.abs(values_before) < threshold)
            small_count = int(mask_small.sum())
            total_small_count += small_count
            
            if small_count > 0:
                chunk_df.loc[:, var_cols] = np.where(mask_small, 0.0, values_before)

            # Drop rows where all variable columns are zero or NaN
            vars_filled = chunk_df[var_cols].fillna(0.0)
            row_nonzero = (np.abs(vars_filled) > 0).any(axis=1)
            chunk_df = chunk_df[row_nonzero].copy()
            
            total_rows_after += len(chunk_df)
            
            # Write chunk to output file
            chunk_df.to_csv(out_path, mode='w' if first_chunk else 'a', 
                           header=first_chunk, index=False)
            first_chunk = False
            
    except Exception as e:
        logger.error(f"  Failed to process CSV chunks: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)
        return csv_path, 0, 0, 0

    # If inplace, replace original file
    if inplace:
        try:
            if os.path.exists(csv_path):
                os.remove(csv_path)
            os.rename(out_path, csv_path)
            out_path = csv_path
        except Exception as e:
            logger.error(f"  Failed to replace original file: {e}")
            return csv_path, total_small_count, 0, 0

    rows_dropped = total_rows_before - total_rows_after
    logger.info(f"  Small values (< {threshold} Mg) set to zero: {total_small_count}")
    logger.info(f"  Rows dropped (all variables zero): {rows_dropped}")
    logger.info(f"  Saved: {out_path}")
    return out_path, total_small_count, rows_dropped, total_rows_after


def clean_file(csv_path: str, threshold: float, id_candidates: List[str], inplace: bool, 
               logger: logging.Logger, chunk_size: Optional[int] = None) -> Tuple[str, int, int, int]:
    """Clean CSV file, optionally using chunked processing for memory efficiency."""
    
    # Use chunked processing if chunk_size is specified
    if chunk_size is not None:
        return clean_file_chunked(csv_path, threshold, id_candidates, inplace, chunk_size, logger)
    
    # Original non-chunked processing for smaller files
    logger.info(f"Processing file: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"  Failed to read CSV: {e}")
        return csv_path, 0, 0, 0

    if df.empty:
        logger.info("  File is empty; skipping")
        return csv_path, 0, 0, 0

    # Identify identifier columns (case-insensitive): default candidates dggsID, GID, Year/year
    id_cols_present = find_id_columns(list(df.columns), id_candidates)
    if not id_cols_present:
        logger.warning("  No identifier columns found; will treat all columns as variables")
    var_cols = [c for c in df.columns if c not in id_cols_present]
    if not var_cols:
        logger.info("  No variable columns to process; skipping")
        return csv_path, 0, 0, 0

    # Ensure numeric for variable columns
    for c in var_cols:
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Count and zero-out values with absolute magnitude < threshold
    values_before = df[var_cols].to_numpy(copy=False)
    mask_small = np.isfinite(values_before) & (np.abs(values_before) < threshold)
    small_count = int(mask_small.sum())
    if small_count > 0:
        df.loc[:, var_cols] = np.where(mask_small, 0.0, values_before)

    # Drop rows where all variable columns are zero or NaN
    # Treat NaN as zero for the purpose of deciding all-zero rows
    vars_filled = df[var_cols].fillna(0.0)
    row_nonzero = (np.abs(vars_filled) > 0).any(axis=1)
    rows_before = len(df)
    df = df[row_nonzero].copy()
    rows_dropped = rows_before - len(df)

    # Output path
    if inplace:
        out_path = csv_path
    else:
        base, ext = os.path.splitext(csv_path)
        out_path = f"{base}_cleaned{ext}"

    try:
        df.to_csv(out_path, index=False)
    except Exception as e:
        logger.error(f"  Failed to write output CSV: {e}")
        return csv_path, small_count, 0, 0

    logger.info(f"  Small values (< {threshold} Mg) set to zero: {small_count}")
    logger.info(f"  Rows dropped (all variables zero): {rows_dropped}")
    logger.info(f"  Saved: {out_path}")
    return out_path, small_count, rows_dropped, len(df)


def main():
    parser = argparse.ArgumentParser(description="Post-process methane CSVs: zero tiny values and drop all-zero rows.")
    parser.add_argument('--path', required=True, help="CSV file or directory containing CSV files")
    parser.add_argument('--threshold', type=float, default=1e-6, help="Threshold in Mg; |value| < threshold -> 0 (default: 1e-6 Mg = 1 g)")
    parser.add_argument('--ids', default='dggsID,GID,Year,year', help="Comma-separated identifier columns to keep untouched")
    parser.add_argument('--inplace', action='store_true', help="Overwrite input files instead of writing *_cleaned.csv")
    parser.add_argument('--chunk-size', type=int, default=10000, help="Chunk size for memory-efficient processing (default: 10000 rows). Set to 0 to disable chunking.")
    args = parser.parse_args()

    logger = setup_logger()

    csv_files = discover_csv_files(args.path)
    if not csv_files:
        logger.error(f"No CSV files found in path: {args.path}")
        sys.exit(1)

    id_candidates = [c.strip() for c in args.ids.split(',') if c.strip()]
    
    # Determine chunk size (0 means no chunking)
    chunk_size = args.chunk_size if args.chunk_size > 0 else None
    if chunk_size:
        logger.info(f"Using chunked processing with chunk size: {chunk_size}")
    else:
        logger.info("Using standard processing (no chunking)")

    total_files = 0
    total_small_set_zero = 0
    total_rows_dropped = 0
    total_rows_remaining = 0

    for f in sorted(csv_files):
        out_path, small_count, rows_dropped, rows_remaining = clean_file(
            f, args.threshold, id_candidates, args.inplace, logger, chunk_size
        )
        total_files += 1
        total_small_set_zero += small_count
        total_rows_dropped += rows_dropped
        total_rows_remaining += rows_remaining

    logger.info("")
    logger.info("=" * 60)
    logger.info("CLEANUP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Files processed: {total_files}")
    logger.info(f"Values set to zero (< {args.threshold} Mg): {total_small_set_zero}")
    logger.info(f"Rows dropped (all variables zero): {total_rows_dropped}")
    logger.info(f"Rows remaining across outputs: {total_rows_remaining}")


if __name__ == '__main__':
    main()


# Memory-efficient version with chunked processing
# cd /home/mingke.li/methane_grid_calculation_ARC
# python scripts/utilities/cleanup_small_values_in_csv.py --path /home/mingke.li/methane_grid_calculation_ARC/output/ --threshold 1e-6 --ids dggsID,GID,Year,year --inplace --chunk-size 10000