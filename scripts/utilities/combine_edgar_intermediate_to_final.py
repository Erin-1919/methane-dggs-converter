import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List, Optional, Set

import pandas as pd


def setup_logger() -> logging.Logger:
    """Configure logger writing to log/ with a timestamped filename and console output."""
    os.makedirs("log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("log", f"combine_edgar_intermediate_to_final_{timestamp}.log")

    logger = logging.getLogger(f"combine_edgar_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s - %(message)s')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    logger.info(f"Logging initialized: {log_path}")
    return logger


def get_sector_folders(test_csv_folder: str) -> List[str]:
    """Return sorted sector folder names under test_csv_folder that end with '_emi_nc'."""
    sectors: List[str] = []
    if not os.path.exists(test_csv_folder):
        return sectors
    for name in os.listdir(test_csv_folder):
        p = os.path.join(test_csv_folder, name)
        if os.path.isdir(p) and name.endswith('_emi_nc'):
            sectors.append(name)
    sectors.sort()
    return sectors


def aggregate_aviation_columns_wide(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Aggregate aviation columns into a single '1A3a' column if they exist, then drop originals."""
    aviation_cols = ['1A3a_CDS', '1A3a_CRS', '1A3a_LTO', '1A3a_SPS']
    present = [c for c in aviation_cols if c in df.columns]
    if not present:
        return df
    logger.info(f"  Aggregating aviation columns into 1A3a: {present}")
    df['1A3a'] = df.get('1A3a', 0.0) + df[present].sum(axis=1)
    df = df.drop(columns=present)
    return df


def melt_and_accumulate(long_accumulator: Optional[pd.DataFrame], df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Melt sector-wide dataframe to long format and accumulate sums by keys."""
    id_cols = ['dggsID', 'GID']
    # Ensure required identifiers exist
    for c in id_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in sector-year dataframe")
    df['Year'] = year
    value_cols = [c for c in df.columns if c not in (id_cols + ['Year'])]
    if not value_cols:
        return long_accumulator if long_accumulator is not None else pd.DataFrame(columns=id_cols + ['Year', 'IPCC', 'value'])
    melted = df.melt(id_vars=id_cols + ['Year'], value_vars=value_cols, var_name='IPCC', value_name='value')
    melted['value'] = melted['value'].fillna(0.0)
    # Early filter zeros to reduce memory
    melted = melted[melted['value'] > 0]
    if long_accumulator is None or long_accumulator.empty:
        return melted
    # Concatenate and periodically reduce by grouping
    combined = pd.concat([long_accumulator, melted], ignore_index=True)
    combined = combined.groupby(['dggsID', 'GID', 'Year', 'IPCC'], as_index=False)['value'].sum()
    return combined


def process_year(test_csv_folder: str, sector_folders: List[str], year: int, logger: logging.Logger) -> Optional[pd.DataFrame]:
    logger.info(f"Processing year {year}")
    long_acc: Optional[pd.DataFrame] = None
    files_found = 0
    for sector in sector_folders:
        sector_path = os.path.join(test_csv_folder, sector)
        fname = f"EDGAR_DGGS_methane_emissions_{sector}_{year}.csv"
        fpath = os.path.join(sector_path, fname)
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            logger.error(f"  Error reading {fpath}: {e}")
            continue
        if df.empty:
            continue
        files_found += 1
        long_acc = melt_and_accumulate(long_acc, df, year)
        logger.info(f"  Added {sector} ({len(df)} rows)")

    if long_acc is None or long_acc.empty:
        logger.warning(f"No data found for year {year}")
        return None

    # Aggregate aviation categories in long form by mapping to 1A3a
    aviation_map = {
        '1A3a_CDS': '1A3a',
        '1A3a_CRS': '1A3a',
        '1A3a_LTO': '1A3a',
        '1A3a_SPS': '1A3a',
    }
    long_acc['IPCC'] = long_acc['IPCC'].map(lambda c: aviation_map.get(c, c))
    long_acc = long_acc.groupby(['dggsID', 'GID', 'Year', 'IPCC'], as_index=False)['value'].sum()

    # Pivot to wide per-year
    id_cols = ['dggsID', 'GID', 'Year']
    wide = long_acc.pivot_table(index=id_cols, columns='IPCC', values='value', aggfunc='sum', fill_value=0.0)
    wide = wide.reset_index()
    # Ensure aviation aggregation complete and originals dropped (mapping already handled)
    wide = aggregate_aviation_columns_wide(wide, logger)
    logger.info(f"  Year {year} wide shape: {wide.shape}")
    return wide


def collect_all_ipcc_columns(per_year_paths: List[str], logger: logging.Logger) -> List[str]:
    """Scan per-year CSV files to build a stable ordered list of IPCC columns."""
    ipcc_set: Set[str] = set()
    for p in per_year_paths:
        try:
            cols = list(pd.read_csv(p, nrows=0).columns)
        except Exception as e:
            logger.error(f"  Error reading header from {p}: {e}")
            continue
        for c in cols:
            if c not in ('dggsID', 'GID', 'Year'):
                ipcc_set.add(c)
    ordered = sorted(ipcc_set)
    logger.info(f"Collected {len(ordered)} IPCC columns across years")
    return ordered


def write_final_all_years(per_year_paths: List[str], output_path: str, ipcc_columns: List[str], logger: logging.Logger) -> None:
    """Append per-year CSVs into a single final CSV with consistent column order."""
    id_cols = ['dggsID', 'GID', 'Year']
    full_cols = id_cols + ipcc_columns
    header_written = False
    # Ensure output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Remove existing final file if present to avoid appending to old content
    if os.path.exists(output_path):
        os.remove(output_path)
    for p in per_year_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            logger.error(f"  Error reading {p}: {e}")
            continue
        # Add any missing IPCC columns with 0.0
        for c in ipcc_columns:
            if c not in df.columns:
                df[c] = 0.0
        # Keep only required columns (drop any unexpected)
        df = df[full_cols]
        df.to_csv(output_path, mode='a', index=False, header=(not header_written))
        header_written = True
        logger.info(f"  Appended {p} -> {output_path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Combine EDGAR intermediate sector-year CSVs into final all-years CSV, memory-efficiently.")
    parser.add_argument('--test_csv_folder', default=os.path.join('test', 'test_EDGAR_csv'), help="Folder containing sector subfolders with intermediate CSVs")
    parser.add_argument('--output_folder', default='output', help="Destination folder for per-year and final combined CSVs")
    parser.add_argument('--start_year', type=int, default=1970, help="Start year inclusive")
    parser.add_argument('--end_year', type=int, default=2022, help="End year inclusive")
    args = parser.parse_args()

    logger = setup_logger()

    test_csv_folder = args.test_csv_folder
    output_folder = args.output_folder
    start_year = args.start_year
    end_year = args.end_year

    if not os.path.exists(test_csv_folder):
        logger.error(f"Intermediate folder not found: {test_csv_folder}")
        sys.exit(1)

    sectors = get_sector_folders(test_csv_folder)
    if not sectors:
        logger.error(f"No sector folders found in {test_csv_folder}")
        sys.exit(1)
    logger.info(f"Found {len(sectors)} sector folders: {', '.join(sectors)}")

    os.makedirs(output_folder, exist_ok=True)
    # Ensure subfolder for per-year EDGAR outputs
    edgar_years_folder = os.path.join(output_folder, "EDGAR")
    os.makedirs(edgar_years_folder, exist_ok=True)

    per_year_paths: List[str] = []
    successful_years: List[int] = []
    failed_years: List[int] = []

    for year in range(start_year, end_year + 1):
        wide = process_year(test_csv_folder, sectors, year, logger)
        if wide is None or wide.empty:
            failed_years.append(year)
            continue
        # Stable column order per-year: id cols then sorted IPCC
        id_cols = ['dggsID', 'GID', 'Year']
        ipcc_cols_sorted = sorted([c for c in wide.columns if c not in id_cols])
        wide = wide[id_cols + ipcc_cols_sorted]
        per_year_output = os.path.join(edgar_years_folder, f"EDGAR_DGGS_methane_emissions_{year}.csv")
        try:
            wide.to_csv(per_year_output, index=False)
            per_year_paths.append(per_year_output)
            successful_years.append(year)
            logger.info(f"Saved per-year CSV: {per_year_output} ({wide.shape})")
        except Exception as e:
            logger.error(f"Error saving per-year CSV for {year}: {e}")
            failed_years.append(year)

    if not per_year_paths:
        logger.error("No per-year CSVs produced; cannot build final file.")
        sys.exit(1)

    # Build final all-years CSV with consistent columns
    ipcc_columns = collect_all_ipcc_columns(per_year_paths, logger)
    final_output = os.path.join(output_folder, f"EDGAR_DGGS_methane_emissions_ALL_SECTORS_{start_year}_{end_year}.csv")
    write_final_all_years(per_year_paths, final_output, ipcc_columns, logger)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMBINATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Years requested: {start_year}-{end_year}")
    logger.info(f"Successful years: {len(successful_years)}")
    logger.info(f"Failed years: {len(failed_years)} -> {failed_years}")
    logger.info(f"Per-year CSVs: {len(per_year_paths)} saved in {output_folder}")
    logger.info(f"Final combined CSV: {final_output}")


if __name__ == '__main__':
    main()


