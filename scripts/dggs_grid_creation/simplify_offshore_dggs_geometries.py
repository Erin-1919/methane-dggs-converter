"""
Simplify DGGS Grid Geometries for Offshore Files.

This script processes individual offshore DGGS grid GeoJSON files and simplifies their geometries
to reduce file size while maintaining spatial accuracy. It preserves all properties and metadata
while applying geometry simplification algorithms.
"""

import os
import json
import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging

# Configuration variables
INPUT_FOLDER = "data/geojson/global_offshore_grid"
OUTPUT_FOLDER = "data/geojson/global_offshore_dggs"
SIMPLIFICATION_TOLERANCE = 0.01  # Degrees - adjust based on needs
MIN_VERTICES = 4  # Minimum number of vertices to keep in a polygon
SKIP_EXISTING = True  # Skip files that already exist in output folder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('offshore_geometry_simplification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OffshoreDGGSGeometrySimplifier:
    """Simplify offshore DGGS grid geometries while preserving spatial integrity."""
    
    def __init__(self, tolerance: float = 0.01, min_vertices: int = 4):
        self.tolerance = tolerance
        self.min_vertices = min_vertices
        
    def simplify_geometry(self, geometry) -> Optional[object]:
        """Simplify a single geometry object."""
        try:
            if geometry is None:
                return None
                
            # Apply simplification with tolerance
            simplified = geometry.simplify(tolerance=self.tolerance, preserve_topology=True)
            
            # Ensure minimum vertex count for polygons
            if hasattr(simplified, 'exterior') and simplified.exterior:
                coords = list(simplified.exterior.coords)
                if len(coords) < self.min_vertices:
                    # If too simplified, use original with minimal simplification
                    simplified = geometry.simplify(tolerance=self.tolerance/10, preserve_topology=True)
                    
            return simplified
            
        except Exception as e:
            logger.error(f"Error simplifying geometry: {e}")
            return geometry  # Return original if simplification fails
    
    def simplify_feature_collection(self, geojson_data: dict, filename: str) -> dict:
        """Simplify all geometries in a FeatureCollection."""
        try:
            simplified_features = []
            
            for feature in geojson_data.get('features', []):
                try:
                    if 'geometry' in feature and feature['geometry']:
                        # Validate geometry structure
                        if not isinstance(feature['geometry'], dict) or 'type' not in feature['geometry']:
                            logger.warning(f"Skipping feature with invalid geometry structure")
                            simplified_features.append(feature)
                            continue
                        
                        # Convert to shapely geometry for simplification
                        try:
                            # Use shapely directly instead of GeoPandas for more reliable conversion
                            from shapely.geometry import shape
                            geom = shape(feature['geometry'])
                        except Exception as geom_error:
                            logger.warning(f"Failed to convert geometry to shapely: {geom_error}")
                            # Try alternative method using GeoPandas with explicit CRS
                            try:
                                import geopandas as gpd
                                geom = gpd.GeoSeries([feature['geometry']], crs="EPSG:4326").iloc[0]
                                logger.info("Successfully converted using GeoPandas fallback")
                            except Exception as fallback_error:
                                logger.warning(f"Fallback conversion also failed: {fallback_error}")
                                simplified_features.append(feature)
                                continue
                        
                        # Simplify geometry
                        simplified_geom = self.simplify_geometry(geom)
                        
                        # Create simplified feature
                        simplified_feature = {
                            "type": "Feature",
                            "properties": feature.get('properties', {}),
                            "geometry": simplified_geom.__geo_interface__
                        }
                        simplified_features.append(simplified_feature)
                    else:
                        # Keep feature as-is if no geometry
                        simplified_features.append(feature)
                        
                except Exception as feature_error:
                    logger.warning(f"Error processing feature: {feature_error}")
                    # Keep the original feature if simplification fails
                    simplified_features.append(feature)
            
            # Extract country_gid and country_name from filename (e.g., "CYPRUS_CYP_offshore_grid" -> "CYP" and "CYPRUS")
            country_gid = self.extract_country_gid_from_filename(filename)
            country_name = self.extract_country_name_from_filename(filename)
            
            # Create simplified GeoJSON with exact same property structure as country files
            simplified_geojson = {
                "type": "FeatureCollection",
                "features": simplified_features,
                "properties": {
                    "country_name": country_name,
                    "country_gid": country_gid,
                    "grid_type": "rhealpix",
                    "resolution": 6,
                    "total_grid_cells": len(simplified_features),
                    "conversion_timestamp": str(pd.Timestamp.now()),
                    "simplification_timestamp": str(pd.Timestamp.now())
                }
            }
            
            return simplified_geojson
            
        except Exception as e:
            logger.error(f"Error simplifying feature collection: {e}")
            return geojson_data
    
    def extract_country_gid_from_filename(self, filename: str) -> str:
        """Extract country_gid from filename (e.g., 'CYPRUS_CYP_offshore_grid' -> 'CYP')."""
        try:
            # Split by underscore and get the second part (index 1)
            parts = filename.split('_')
            if len(parts) >= 2:
                return parts[1]
            else:
                logger.warning(f"Could not extract country_gid from filename: {filename}")
                return "UNK"  # Unknown
        except Exception as e:
            logger.error(f"Error extracting country_gid from filename {filename}: {e}")
            return "UNK"
    
    def extract_country_name_from_filename(self, filename: str) -> str:
        """Extract country name from filename (e.g., 'CYPRUS_CYP_offshore_grid' -> 'CYPRUS')."""
        try:
            # Split by underscore and get the first part (index 0)
            parts = filename.split('_')
            if len(parts) >= 1:
                return parts[0]
            else:
                logger.warning(f"Could not extract country name from filename: {filename}")
                return "UNKNOWN"  # Unknown
        except Exception as e:
            logger.error(f"Error extracting country name from filename {filename}: {e}")
            return "UNKNOWN"
    
    def get_file_size_mb(self, filepath: str) -> float:
        """Get file size in megabytes."""
        try:
            size_bytes = os.path.getsize(filepath)
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0.0
    
    def process_single_file(self, input_filepath: str, output_filepath: str) -> Tuple[bool, dict]:
        """Process a single offshore GeoJSON file."""
        try:
            # Get original file size
            original_size = self.get_file_size_mb(input_filepath)
            
            # Load GeoJSON
            logger.info(f"Loading file: {os.path.basename(input_filepath)}")
            with open(input_filepath, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            # Simplify geometries
            filename = os.path.basename(input_filepath)
            logger.info(f"Simplifying geometries for {filename}")
            simplified_geojson = self.simplify_feature_collection(geojson_data, filename)
            
            # Save simplified file
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(simplified_geojson, f, indent=2)
            
            # Get simplified file size
            simplified_size = self.get_file_size_mb(output_filepath)
            
            # Calculate statistics
            stats = {
                'original_size_mb': round(original_size, 2),
                'simplified_size_mb': round(simplified_size, 2),
                'size_reduction_mb': round(original_size - simplified_size, 2),
                'size_reduction_percent': round(
                    ((original_size - simplified_size) / original_size * 100) if original_size > 0 else 0, 1
                )
            }
            
            logger.info(f"Successfully processed {os.path.basename(input_filepath)}")
            logger.info(f"  Size: {stats['original_size_mb']}MB -> {stats['simplified_size_mb']}MB "
                    f"({stats['size_reduction_percent']}% reduction)")
            
            return True, stats
            
        except Exception as e:
            logger.error(f"Error processing {input_filepath}: {e}")
            return False, {}
    
    def process_all_files(self, input_folder: str, output_folder: str) -> bool:
        """Process all offshore GeoJSON files in the input folder."""
        try:
            input_path = Path(input_folder)
            output_path = Path(output_folder)
            
            if not input_path.exists():
                logger.error(f"Input folder does not exist: {input_folder}")
                return False
            
            # Create output folder
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Find all GeoJSON files
            geojson_files = list(input_path.glob("*.geojson"))
            
            if not geojson_files:
                logger.warning(f"No GeoJSON files found in {input_folder}")
                return False
            
            logger.info(f"Found {len(geojson_files)} offshore GeoJSON files to process")
            logger.info(f"Input folder: {input_folder}")
            logger.info(f"Output folder: {output_folder}")
            logger.info(f"Simplification tolerance: {self.tolerance}")
            logger.info(f"Minimum vertices: {self.min_vertices}")
            
            # Process each file
            successful_files = 0
            failed_files = 0
            total_original_size = 0
            total_simplified_size = 0
            
            for input_file in geojson_files:
                output_file = output_path / input_file.name
                
                # Skip if file already exists and SKIP_EXISTING is True
                if SKIP_EXISTING and output_file.exists():
                    logger.info(f"Skipping {input_file.name} (already exists)")
                    continue
                
                success, stats = self.process_single_file(str(input_file), str(output_file))
                
                if success:
                    successful_files += 1
                    total_original_size += stats.get('original_size_mb', 0)
                    total_simplified_size += stats.get('simplified_size_mb', 0)
                else:
                    failed_files += 1
            
            # Summary statistics
            logger.info("\n" + "="*60)
            logger.info("OFFSHORE PROCESSING SUMMARY")
            logger.info("="*60)
            logger.info(f"Total files processed: {successful_files + failed_files}")
            logger.info(f"Successful: {successful_files}")
            logger.info(f"Failed: {failed_files}")
            logger.info(f"Total original size: {round(total_original_size, 2)}MB")
            logger.info(f"Total simplified size: {round(total_simplified_size, 2)}MB")
            logger.info(f"Total size reduction: {round(total_original_size - total_simplified_size, 2)}MB")
            if total_original_size > 0:
                total_reduction_percent = round(
                    ((total_original_size - total_simplified_size) / total_original_size * 100), 1
                )
                logger.info(f"Overall size reduction: {total_reduction_percent}%")
            
            return failed_files == 0
            
        except Exception as e:
            logger.error(f"Error processing files: {e}")
            return False


def main():
    """Main function to run the offshore geometry simplification."""
    logger.info("Starting Offshore DGGS Geometry Simplification Process")
    
    # Validate input folder
    if not os.path.exists(INPUT_FOLDER):
        logger.error(f"Input folder not found: {INPUT_FOLDER}")
        logger.error("Please ensure the input folder exists and contains GeoJSON files")
        return
    
    # Create simplifier instance
    simplifier = OffshoreDGGSGeometrySimplifier(
        tolerance=SIMPLIFICATION_TOLERANCE,
        min_vertices=MIN_VERTICES
    )
    
    # Process all files
    success = simplifier.process_all_files(INPUT_FOLDER, OUTPUT_FOLDER)
    
    if success:
        logger.info("\nOffshore geometry simplification completed successfully!")
        logger.info(f"Simplified files saved to: {OUTPUT_FOLDER}")
    else:
        logger.error("\nOffshore geometry simplification completed with errors!")
        logger.error("Check the log file for details")
        exit(1)


if __name__ == "__main__":
    main()
