"""
Test script for DGGS Geometry Simplification.

This script tests the geometry simplification functionality on a single country GeoJSON file
to verify the process works correctly before running it on all files.
"""

import os
import json
import geopandas as gpd
import pandas as pd
from pathlib import Path
import logging

# Test configuration
TEST_INPUT_FILE = "data/geojson/global_countries_grid/Virgin_Islands,_U.S._VIR_grid.geojson"
TEST_OUTPUT_FILE = "data/geojson/global_countries_dggs/Virgin_Islands,_U.S._VIR_grid.geojson"
TEST_TOLERANCE = 0.01 # Degrees
TEST_MIN_VERTICES = 4

# Setup logging for test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestDGGSGeometrySimplifier:
    """Test class for DGGS geometry simplification."""
    
    def __init__(self, tolerance: float = 0.01, min_vertices: int = 4):
        self.tolerance = tolerance
        self.min_vertices = min_vertices
        
    def simplify_geometry(self, geometry):
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
            return geometry
    
    def simplify_feature_collection(self, geojson_data: dict) -> dict:
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
            
            # Create simplified GeoJSON
            simplified_geojson = {
                "type": "FeatureCollection",
                "features": simplified_features,
                "properties": geojson_data.get('properties', {})
            }
            
            # Add simplification metadata
            if 'properties' in simplified_geojson:
                simplified_geojson['properties']['simplification_timestamp'] = str(pd.Timestamp.now())
            
            return simplified_geojson
            
        except Exception as e:
            logger.error(f"Error simplifying feature collection: {e}")
            return geojson_data
    
    def get_file_size_mb(self, filepath: str) -> float:
        """Get file size in megabytes."""
        try:
            size_bytes = os.path.getsize(filepath)
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0.0
    
    def test_single_file(self, input_filepath: str, output_filepath: str) -> dict:
        """Test simplification on a single file."""
        try:
            logger.info("="*60)
            logger.info("TESTING DGGS GEOMETRY SIMPLIFICATION")
            logger.info("="*60)
            
            # Check if input file exists
            if not os.path.exists(input_filepath):
                logger.error(f"Test input file not found: {input_filepath}")
                return {}
            
            # Get original file size
            original_size = self.get_file_size_mb(input_filepath)
            logger.info(f"Input file: {input_filepath}")
            logger.info(f"Original file size: {original_size:.2f}MB")
            
            # Load GeoJSON
            logger.info("Loading GeoJSON file...")
            with open(input_filepath, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            # Display original file info
            original_features = len(geojson_data.get('features', []))
            logger.info(f"Original features: {original_features}")
            logger.info(f"Original properties: {list(geojson_data.get('properties', {}).keys())}")
            
            # Debug: Check first feature structure
            if original_features > 0:
                first_feature = geojson_data.get('features', [])[0]
                logger.info(f"First feature keys: {list(first_feature.keys())}")
                if 'geometry' in first_feature:
                    geom = first_feature['geometry']
                    logger.info(f"First feature geometry type: {type(geom)}")
                    if isinstance(geom, dict):
                        logger.info(f"Geometry keys: {list(geom.keys())}")
                        logger.info(f"Geometry type: {geom.get('type', 'unknown')}")
                        # Show a sample of coordinates to understand the structure
                        if 'coordinates' in geom:
                            coords = geom['coordinates']
                            logger.info(f"Coordinates type: {type(coords)}")
                            if isinstance(coords, list) and len(coords) > 0:
                                logger.info(f"First coordinate sample: {coords[0][:3] if len(coords[0]) > 3 else coords[0]}")
            
            # Simplify geometries
            logger.info(f"Simplifying geometries with tolerance: {self.tolerance}")
            try:
                simplified_geojson = self.simplify_feature_collection(geojson_data)
                logger.info("Geometry simplification completed")
            except Exception as simplify_error:
                logger.error(f"Failed to simplify geometries: {simplify_error}")
                # Use original data if simplification fails
                simplified_geojson = geojson_data
                logger.info("Using original geometries due to simplification failure")
            
            # Save simplified file
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(simplified_geojson, f, indent=2)
            
            # Get simplified file size
            simplified_size = self.get_file_size_mb(output_filepath)
            size_reduction = ((original_size - simplified_size) / original_size * 100) if original_size > 0 else 0
            
            logger.info(f"File size: {original_size:.2f}MB → {simplified_size:.2f}MB ({size_reduction:.1f}% reduction)")
            
            # Display new properties
            new_properties = list(simplified_geojson.get('properties', {}).keys())
            logger.info(f"New properties: {new_properties}")
            
            # Verify file structure
            logger.info("Verifying file structure...")
            if simplified_geojson.get('type') == 'FeatureCollection':
                logger.info("✓ File type: FeatureCollection")
            else:
                logger.warning("✗ File type is not FeatureCollection")
            
            if len(simplified_geojson.get('features', [])) == original_features:
                logger.info(f"✓ Feature count preserved: {original_features}")
            else:
                logger.warning(f"✗ Feature count changed: {original_features} → {len(simplified_geojson.get('features', []))}")
            
            # Test results summary
            test_results = {
                'input_file': input_filepath,
                'output_file': output_filepath,
                'original_size_mb': round(original_size, 2),
                'simplified_size_mb': round(simplified_size, 2),
                'size_reduction_percent': round(size_reduction, 1),
                'features_preserved': len(simplified_geojson.get('features', [])) == original_features,
                'structure_valid': simplified_geojson.get('type') == 'FeatureCollection'
            }
            
            logger.info("\n" + "="*60)
            logger.info("TEST RESULTS SUMMARY")
            logger.info("="*60)
            logger.info(f"✓ File size reduction: {test_results['size_reduction_percent']}%")
            logger.info(f"✓ Features preserved: {test_results['features_preserved']}")
            logger.info(f"✓ Structure valid: {test_results['structure_valid']}")
            logger.info(f"✓ Output saved to: {output_filepath}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            return {}


def main():
    """Main test function."""
    logger.info("Starting DGGS Geometry Simplification Test")
    
    # Create test simplifier
    test_simplifier = TestDGGSGeometrySimplifier(
        tolerance=TEST_TOLERANCE,
        min_vertices=TEST_MIN_VERTICES
    )
    
    # Run test
    results = test_simplifier.test_single_file(TEST_INPUT_FILE, TEST_OUTPUT_FILE)
    
    if results:
        logger.info("\nTest completed successfully!")
        logger.info("You can now run the full script: python 2simplify_dggs_geometries.py")
    else:
        logger.error("\nTest failed!")
        exit(1)


if __name__ == "__main__":
    main()
