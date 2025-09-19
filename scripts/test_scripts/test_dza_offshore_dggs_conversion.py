"""
Test script to convert Algeria offshore facilities (GID=DZA) to DGGS grid and save as shapefile.
This script tests the offshore DGGS conversion functionality.
"""

import os
import sys
import json
import geopandas as gpd
import pandas as pd
from pathlib import Path

# Add the scripts directory to the path so we can import the converter
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dggs_grid_creation'))

from convert_offshore_to_dggs import OffshoreToDGGSConverter


def test_dza_offshore_dggs_conversion():
    """Test DGGS conversion for Algeria offshore facilities (GID=DZA) and save as shapefile."""
    
    # Configuration
    input_file = "data/geojson/global_offshore_cleaned.geojson"
    output_folder = "data/shp"
    target_gid = "DZA"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Loading offshore GeoJSON file: {input_file}")
    
    # Load the GeoJSON file
    gdf = gpd.read_file(input_file)
    print(f"Loaded {len(gdf)} features from {input_file}")
    print(f"CRS: {gdf.crs}")
    print(f"Columns: {list(gdf.columns)}")
    
    # Find the Algeria offshore feature
    dza_feature = gdf[gdf['GID'] == target_gid]
    
    if dza_feature.empty:
        print(f"Error: No feature found with GID={target_gid}")
        return False
    
    print(f"\nFound Algeria offshore feature:")
    print(f"  NAME: {dza_feature.iloc[0]['NAME']}")
    print(f"  GID: {dza_feature.iloc[0]['GID']}")
    print(f"  Geometry type: {dza_feature.iloc[0].geometry.geom_type}")
    
    # Get the geometry
    dza_geometry = dza_feature.iloc[0].geometry
    dza_name = dza_feature.iloc[0]['NAME']
    dza_gid = dza_feature.iloc[0]['GID']
    
    # Initialize the converter
    converter = OffshoreToDGGSConverter(dggs_grid_type="rhealpix", resolution=6)
    
    print(f"\nConverting Algeria offshore facilities to DGGS grid...")
    print(f"Grid type: rhealpix")
    print(f"Resolution: 6")
    
    # Convert to DGGS grid
    dggs_features = converter.convert_offshore_facility_to_dggs_grid(dza_geometry, dza_name, dza_gid)
    
    if not dggs_features:
        print("Error: No DGGS grid cells generated for Algeria offshore facilities")
        return False
    
    print(f"Generated {len(dggs_features)} DGGS grid cells for Algeria offshore facilities")
    
    # Create GeoDataFrame from the DGGS features
    dggs_gdf = gpd.GeoDataFrame.from_features(dggs_features)
    dggs_gdf = dggs_gdf.set_crs("EPSG:4326")
    
    print(f"DGGS GeoDataFrame shape: {dggs_gdf.shape}")
    print(f"DGGS GeoDataFrame columns: {list(dggs_gdf.columns)}")
    
    # Save as shapefile
    output_shapefile = os.path.join(output_folder, f"algeria_offshore_dggs_grid.shp")
    
    print(f"\nSaving DGGS grid as shapefile: {output_shapefile}")
    dggs_gdf.to_file(output_shapefile)
    
    # Also save as GeoJSON for comparison
    output_geojson = os.path.join(output_folder, f"algeria_offshore_dggs_grid.geojson")
    dggs_gdf.to_file(output_geojson, driver="GeoJSON")
    
    print(f"Also saved as GeoJSON: {output_geojson}")
    
    # Print some statistics
    print(f"\nConversion completed successfully!")
    print(f"Total DGGS grid cells: {len(dggs_features)}")
    print(f"Output files:")
    print(f"  - Shapefile: {output_shapefile}")
    print(f"  - GeoJSON: {output_geojson}")
    
    # Print first few features for verification
    print(f"\nFirst 3 DGGS grid cells:")
    for i, feature in enumerate(dggs_features[:3]):
        print(f"  {i+1}. ZoneID: {feature['properties']['zoneID']}")
        print(f"     NAME: {feature['properties']['NAME']}")
        print(f"     GID: {feature['properties']['GID']}")
    
    # Print geometry bounds for verification
    print(f"\nAlgeria offshore geometry bounds:")
    bounds = dza_geometry.bounds
    print(f"  Min X: {bounds[0]:.6f}")
    print(f"  Min Y: {bounds[1]:.6f}")
    print(f"  Max X: {bounds[2]:.6f}")
    print(f"  Max Y: {bounds[3]:.6f}")
    
    return True


def main():
    """Main function to run the test."""
    print("Testing DGGS conversion for Algeria offshore facilities (GID=DZA)")
    print("=" * 70)
    
    success = test_dza_offshore_dggs_conversion()
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
