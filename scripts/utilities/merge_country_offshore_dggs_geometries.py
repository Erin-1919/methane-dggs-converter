"""
Script to merge offshore and onshore geospatial data by combining features from 
offshore geojson files into corresponding country geojson files.

This script:
1. Reads offshore geojson files from global_offshore_dggs/
2. Reads country geojson files from global_countries_dggs/
3. Matches files by country GID (country abbreviation)
4. Merges features from offshore files into country files
5. Prevents duplicate features
6. Maintains data consistency and naming conventions
7. Outputs merged files to global_countries_dggs_merge/
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeojsonMerger:
    """Class to handle merging of offshore and onshore geojson files."""
    
    def __init__(self, offshore_dir: str, countries_dir: str, output_dir: str):
        """
        Initialize the merger with directory paths.
        
        Args:
            offshore_dir: Path to offshore geojson files
            countries_dir: Path to country geojson files
            output_dir: Path to output merged files
        """
        self.offshore_dir = Path(offshore_dir)
        self.countries_dir = Path(countries_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for country files to avoid repeated searches
        self.country_files_cache = {}
        
    def extract_gid_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract country GID from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Country GID if found, None otherwise
        """
        # Remove file extension
        name_without_ext = filename.replace('.geojson', '')
        
        # For offshore files, look for GID that appears after an underscore
        # This helps distinguish between country names and GIDs
        if '_' in name_without_ext:
            parts = name_without_ext.split('_')
            for part in parts:
                if len(part) == 3 and part.isupper() and part.isalpha():
                    # Check if this part looks like a country GID
                    # Most country GIDs are not common words
                    if part not in ['THE', 'AND', 'FOR', 'ARE', 'USA', 'UK']:
                        return part
        
        # Fallback: look for 3-letter country codes (GIDs) using regex
        # Common country GIDs are 3 uppercase letters
        gid_pattern = r'[A-Z]{3}'
        matches = re.findall(gid_pattern, name_without_ext)
        
        if matches:
            # If multiple matches, try to find the most likely GID
            if len(matches) > 1:
                # Look for a GID that's surrounded by underscores or at boundaries
                for match in matches:
                    # Check if this 3-letter code is surrounded by underscores
                    if f'_{match}_' in name_without_ext:
                        return match
                    # Check if it's at the beginning or end with underscore
                    if name_without_ext.startswith(f'{match}_') or name_without_ext.endswith(f'_{match}'):
                        return match
                    # Check if it's followed by 'offshore' or 'grid' (common patterns)
                    if f'_{match}_offshore' in name_without_ext or f'_{match}_grid' in name_without_ext:
                        return match
            
            # Return the first match if no better option found
            return matches[0]
        
        return None
    
    def build_country_files_index(self) -> Dict[str, str]:
        """
        Build an index of country files by GID for efficient lookup.
        
        Returns:
            Dictionary mapping GID to country filename
        """
        logger.info("Building country files index...")
        index = {}
        
        for file_path in self.countries_dir.glob('*.geojson'):
            gid = self.extract_gid_from_filename(file_path.name)
            if gid:
                index[gid] = file_path.name
                logger.debug(f"Mapped GID {gid} to file {file_path.name}")
        
        logger.info(f"Indexed {len(index)} country files")
        return index
    
    def load_geojson_file(self, file_path: Path) -> Dict:
        """
        Load and parse a geojson file.
        
        Args:
            file_path: Path to the geojson file
            
        Returns:
            Parsed geojson data as dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def save_geojson_file(self, data: Dict, file_path: Path) -> bool:
        """
        Save geojson data to file.
        
        Args:
            data: Geojson data to save
            file_path: Path where to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")
            return False
    
    def merge_features(self, country_data: Dict, offshore_data: Dict) -> Dict:
        """
        Merge features from offshore data into country data.
        
        Args:
            country_data: Country geojson data
            offshore_data: Offshore geojson data
            
        Returns:
            Merged geojson data
        """
        if not offshore_data or 'features' not in offshore_data:
            return country_data
        
        # Get existing features
        merged_features = country_data.get('features', [])
        
        # Add offshore features, avoiding duplicates
        offshore_features = offshore_data.get('features', [])
        
        # Create a set of existing zoneIDs to avoid duplicates
        existing_zone_ids = set()
        for feature in merged_features:
            zone_id = feature.get('properties', {}).get('zoneID')
            if zone_id:
                existing_zone_ids.add(zone_id)
        
        # Add offshore features that don't have duplicate zoneIDs
        added_count = 0
        for feature in offshore_features:
            zone_id = feature.get('properties', {}).get('zoneID')
            if zone_id and zone_id not in existing_zone_ids:
                merged_features.append(feature)
                existing_zone_ids.add(zone_id)
                added_count += 1
        
        logger.info(f"Added {added_count} offshore features")
        
        # Update the merged data
        merged_data = country_data.copy()
        merged_data['features'] = merged_features
        
        # Update total grid cells count if it exists
        if 'properties' in merged_data and 'total_grid_cells' in merged_data['properties']:
            merged_data['properties']['total_grid_cells'] = len(merged_features)
        
        return merged_data
    
    def process_offshore_file(self, offshore_file: Path, country_files_index: Dict[str, str]) -> bool:
        """
        Process a single offshore file and merge it with corresponding country file.
        
        Args:
            offshore_file: Path to offshore geojson file
            country_files_index: Index of country files by GID
            
        Returns:
            True if successful, False otherwise
        """
        # Extract GID from offshore filename
        gid = self.extract_gid_from_filename(offshore_file.name)
        if not gid:
            logger.warning(f"Could not extract GID from {offshore_file.name}")
            return False
        
        # Find corresponding country file
        country_filename = country_files_index.get(gid)
        if not country_filename:
            logger.warning(f"No country file found for GID {gid} from {offshore_file.name}")
            return False
        
        logger.info(f"Processing {offshore_file.name} -> {country_filename}")
        
        # Load both files
        offshore_data = self.load_geojson_file(offshore_file)
        country_file_path = self.countries_dir / country_filename
        country_data = self.load_geojson_file(country_file_path)
        
        if not offshore_data or not country_data:
            logger.error(f"Failed to load data for {offshore_file.name} or {country_filename}")
            return False
        
        # Merge the data
        merged_data = self.merge_features(country_data, offshore_data)
        
        # Save merged file
        output_file = self.output_dir / country_filename
        if self.save_geojson_file(merged_data, output_file):
            logger.info(f"Successfully merged and saved {output_file}")
            return True
        else:
            logger.error(f"Failed to save merged file {output_file}")
            return False
    
    def run_merge(self) -> None:
        """
        Run the complete merge process for all offshore files.
        """
        logger.info("Starting geojson merge process...")
        
        # Build country files index
        country_files_index = self.build_country_files_index()
        
        if not country_files_index:
            logger.error("No country files found. Cannot proceed.")
            return
        
        # Process each offshore file
        offshore_files = list(self.offshore_dir.glob('*.geojson'))
        logger.info(f"Found {len(offshore_files)} offshore files to process")
        
        successful_merges = 0
        failed_merges = 0
        
        for offshore_file in offshore_files:
            if self.process_offshore_file(offshore_file, country_files_index):
                successful_merges += 1
            else:
                failed_merges += 1
        
        logger.info(f"Merge process completed. Successful: {successful_merges}, Failed: {failed_merges}")
        
        # Copy remaining country files that don't have offshore data
        self.copy_remaining_country_files(country_files_index)
    
    def copy_remaining_country_files(self, country_files_index: Dict[str, str]) -> None:
        """
        Copy country files that don't have corresponding offshore data.
        
        Args:
            country_files_index: Index of country files by GID
        """
        logger.info("Copying remaining country files without offshore data...")
        
        # Get list of already processed country files
        processed_files = set()
        for output_file in self.output_dir.glob('*.geojson'):
            processed_files.add(output_file.name)
        
        # Copy unprocessed country files
        copied_count = 0
        for gid, filename in country_files_index.items():
            if filename not in processed_files:
                source_file = self.countries_dir / filename
                target_file = self.output_dir / filename
                
                if source_file.exists():
                    try:
                        import shutil
                        shutil.copy2(source_file, target_file)
                        copied_count += 1
                        logger.debug(f"Copied {filename}")
                    except Exception as e:
                        logger.error(f"Failed to copy {filename}: {e}")
        
        logger.info(f"Copied {copied_count} remaining country files")


def main():
    """Main function to run the geojson merger."""
    # Set paths - modify these as needed
    offshore_dir = 'data/geojson/global_offshore_dggs'
    countries_dir = 'data/geojson/global_countries_dggs'
    output_dir = 'data/geojson/global_countries_dggs_merge'
    
    # Enable verbose logging if needed
    # logging.getLogger().setLevel(logging.DEBUG)
    
    # Create merger and run
    merger = GeojsonMerger(offshore_dir, countries_dir, output_dir)
    merger.run_merge()


if __name__ == '__main__':
    main()
