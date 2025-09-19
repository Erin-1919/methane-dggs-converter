"""
Test script to convert Faroe Islands (GID=FRO) to DGGS grid and save as shapefile.
This script tests the modified intersection-based filtering in convert_country_geojson_to_dggs.py
"""

import os
import sys
import json
import geopandas as gpd
import pandas as pd
from pathlib import Path

# Add the scripts directory to the path so we can import the converter
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'dggs_grid_creation'))

from convert_country_geojson_to_dggs import GeoJSONToDGGSConverter


def test_fro_dggs_conversion():
    """Test DGGS conversion for Faroe Islands (GID=FRO) and save as shapefile."""
    
    # Configuration
    input_file = "data/geojson/global_countries_simplify.geojson"
    output_folder = "data/shp"
    target_gid = "FRO"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Loading GeoJSON file: {input_file}")
    
    # Load the GeoJSON file
    gdf = gpd.read_file(input_file)
    print(f"Loaded {len(gdf)} features from {input_file}")
    print(f"CRS: {gdf.crs}")
    print(f"Columns: {list(gdf.columns)}")
    
    # Find the Faroe Islands feature
    fro_feature = gdf[gdf['GID'] == target_gid]
    
    if fro_feature.empty:
        print(f"Error: No feature found with GID={target_gid}")
        return False
    
    print(f"\nFound Faroe Islands feature:")
    print(f"  NAME: {fro_feature.iloc[0]['NAME']}")
    print(f"  GID: {fro_feature.iloc[0]['GID']}")
    print(f"  Geometry type: {fro_feature.iloc[0].geometry.geom_type}")
    
    # Get the geometry
    fro_geometry = fro_feature.iloc[0].geometry
    fro_name = fro_feature.iloc[0]['NAME']
    fro_gid = fro_feature.iloc[0]['GID']
    
    # Initialize the converter
    converter = GeoJSONToDGGSConverter(dggs_grid_type="rhealpix", resolution=6)
    
    print(f"\nConverting Faroe Islands to DGGS grid...")
    print(f"Grid type: rhealpix")
    print(f"Resolution: 6")
    
    # Convert to DGGS grid
    dggs_features = converter.convert_country_to_dggs_grid(fro_geometry, fro_name, fro_gid)
    
    if not dggs_features:
        print("Error: No DGGS grid cells generated for Faroe Islands")
        return False
    
    print(f"Generated {len(dggs_features)} DGGS grid cells for Faroe Islands")
    
    # Create GeoDataFrame from the DGGS features
    dggs_gdf = gpd.GeoDataFrame.from_features(dggs_features)
    dggs_gdf = dggs_gdf.set_crs("EPSG:4326")
    
    print(f"DGGS GeoDataFrame shape: {dggs_gdf.shape}")
    print(f"DGGS GeoDataFrame columns: {list(dggs_gdf.columns)}")
    
    # Save as shapefile
    output_shapefile = os.path.join(output_folder, f"faroes_islands_dggs_grid.shp")
    
    print(f"\nSaving DGGS grid as shapefile: {output_shapefile}")
    dggs_gdf.to_file(output_shapefile)
    
    # Also save as GeoJSON for comparison
    output_geojson = os.path.join(output_folder, f"faroes_islands_dggs_grid.geojson")
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
    
    return True


def main():
    """Main function to run the test."""
    print("Testing DGGS conversion for Faroe Islands (GID=FRO)")
    print("=" * 60)
    
    success = test_fro_dggs_conversion()
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
