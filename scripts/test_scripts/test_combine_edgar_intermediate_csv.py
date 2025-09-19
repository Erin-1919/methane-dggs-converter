import os
import pandas as pd
import logging
from datetime import datetime
from collections import defaultdict


def setup_logger():
    """Setup logging configuration"""
    log_folder = "log"
    os.makedirs(log_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"combine_edgar_intermediate_csv_{timestamp}.log"
    log_path = os.path.join(log_folder, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), log_path


def get_sector_folders(test_csv_folder):
    """Get all sector folders from the test CSV directory"""
    sector_folders = []
    for item in os.listdir(test_csv_folder):
        item_path = os.path.join(test_csv_folder, item)
        if os.path.isdir(item_path) and item.endswith('_emi_nc'):
            sector_folders.append(item)
    sector_folders.sort()
    return sector_folders


def aggregate_aviation_columns(df):
    """Aggregate the four aviation columns into a single 1A3a column"""
    logger = logging.getLogger(__name__)
    
    # Check if the aviation columns exist
    aviation_columns = ['1A3a_CDS', '1A3a_CRS', '1A3a_LTO', '1A3a_SPS']
    existing_aviation_columns = [col for col in aviation_columns if col in df.columns]
    
    if not existing_aviation_columns:
        logger.info("  No aviation columns found to aggregate")
        return df
    
    logger.info(f"  Found aviation columns: {existing_aviation_columns}")
    
    # Create the new 1A3a column by summing the existing aviation columns
    df['1A3a'] = df[existing_aviation_columns].sum(axis=1)
    
    # Drop the individual aviation columns
    df = df.drop(columns=existing_aviation_columns)
    
    logger.info(f"  Aggregated {len(existing_aviation_columns)} aviation columns into new 1A3a column")
    logger.info(f"  Remaining columns: {list(df.columns)}")
    
    return df


def process_year_data(test_csv_folder, year, sector_folders):
    """Process all sectors for a specific year and create a combined CSV for that year"""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing year {year}")
    
    all_sector_data = []
    
    for sector_folder in sector_folders:
        try:
            # Check if this sector has data for this year
            sector_path = os.path.join(test_csv_folder, sector_folder)
            year_filename = f"EDGAR_DGGS_methane_emissions_{sector_folder}_{year}.csv"
            year_file_path = os.path.join(sector_path, year_filename)
            
            if os.path.exists(year_file_path):
                logger.info(f"  Reading {sector_folder} for year {year}")
                
                # Read the sector-year file
                df = pd.read_csv(year_file_path)
                
                # Ensure Year column is present and correct
                if 'Year' not in df.columns:
                    df['Year'] = year
                else:
                    df['Year'] = df['Year'].fillna(year)
                
                # Filter out rows where all IPCC columns are 0
                ipcc_columns = [col for col in df.columns if col not in ['dggsID', 'GID', 'Year']]
                if ipcc_columns:
                    # Keep rows where at least one IPCC column has a value > 0
                    df = df[df[ipcc_columns].sum(axis=1) > 0]
                
                if len(df) > 0:
                    all_sector_data.append(df)
                    logger.info(f"    {sector_folder}: {len(df)} records")
                else:
                    logger.info(f"    {sector_folder}: No non-zero records")
            else:
                logger.info(f"  No data for {sector_folder} in year {year}")
                
        except Exception as e:
            logger.error(f"  Error processing {sector_folder} for year {year}: {e}")
    
    if not all_sector_data:
        logger.warning(f"No data found for year {year}")
        return None
    
    logger.info(f"Combining {len(all_sector_data)} sectors for year {year}")
    
    # Combine all sectors for this year
    try:
        combined_df = pd.concat(all_sector_data, ignore_index=True)
        logger.info(f"Combined dataframe for year {year}: {combined_df.shape}")
    except Exception as e:
        logger.error(f"Error combining sectors for year {year}: {e}")
        return None
    
    # Ensure Year column is correct
    combined_df['Year'] = year
    
    # Get the value columns (IPCC codes)
    id_cols = ['dggsID', 'GID', 'Year']
    value_columns = [c for c in combined_df.columns if c not in id_cols]
    
    if not value_columns:
        logger.warning(f"No value columns found for year {year}")
        return None
    
    # Convert to long format to eliminate duplicates
    long_df = combined_df.melt(
        id_vars=id_cols, 
        value_vars=value_columns, 
        var_name='IPCC', 
        value_name='value'
    )
    
    # Fill NaN values with 0
    long_df['value'] = long_df['value'].fillna(0.0)
    
    # Group by ID columns and IPCC, sum values
    long_df = long_df.groupby(id_cols + ['IPCC'], as_index=False)['value'].sum()
    
    # Filter out rows with 0 values
    long_df = long_df[long_df['value'] > 0]
    
    # Convert back to wide format
    wide_df = long_df.pivot_table(
        index=id_cols, 
        columns='IPCC', 
        values='value', 
        aggfunc='sum', 
        fill_value=0.0
    )
    
    # Reset index to get columns back
    wide_df = wide_df.reset_index()
    
    # Get IPCC columns and sort them
    ipcc_cols = sorted([c for c in wide_df.columns if c not in id_cols])
    
    # Reorder columns: ID columns first, then IPCC columns
    wide_df = wide_df[id_cols + ipcc_cols]
    
    # Aggregate aviation columns if they exist
    wide_df = aggregate_aviation_columns(wide_df)
    
    # Save the year-specific CSV file in the same directory
    output_filename = f"EDGAR_DGGS_methane_emissions_{year}.csv"
    output_path = os.path.join(test_csv_folder, output_filename)
    
    wide_df.to_csv(output_path, index=False)
    
    logger.info(f"Saved year {year} CSV: {output_path} with shape {wide_df.shape}")
    return output_path


def main():
    """Main function to create separate CSV files for each year"""
    logger, log_path = setup_logger()
    
    logger.info("Starting EDGAR intermediate CSV combination process by year")
    
    # Define paths
    test_csv_folder = "test/test_EDGAR_csv"
    
    if not os.path.exists(test_csv_folder):
        logger.error(f"Test CSV folder not found: {test_csv_folder}")
        return
    
    logger.info(f"Test CSV folder: {test_csv_folder}")
    
    # Get all sector folders
    sector_folders = get_sector_folders(test_csv_folder)
    logger.info(f"Found {len(sector_folders)} sector folders")
    
    # Process each year from 1970 to 2022
    start_year = 1970
    end_year = 2022
    successful_years = []
    failed_years = []
    
    logger.info(f"Processing years {start_year} to {end_year}")
    
    for year in range(start_year, end_year + 1):
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"PROCESSING YEAR {year}")
            logger.info(f"{'='*50}")
            
            output_path = process_year_data(test_csv_folder, year, sector_folders)
            
            if output_path:
                successful_years.append(year)
                logger.info(f"Year {year} completed successfully")
            else:
                failed_years.append(year)
                logger.warning(f"Year {year} failed")
                
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}")
            failed_years.append(year)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Year range: {start_year} to {end_year}")
    logger.info(f"Successful years: {len(successful_years)}")
    logger.info(f"Failed years: {len(failed_years)}")
    logger.info(f"Successful years: {sorted(successful_years)}")
    if failed_years:
        logger.info(f"Failed years: {sorted(failed_years)}")
    
    logger.info(f"\nOutput files saved to: {test_csv_folder}")
    logger.info("Each year has its own CSV file with aggregated aviation columns")


if __name__ == "__main__":
    main()
