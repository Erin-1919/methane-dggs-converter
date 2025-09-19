from pathlib import Path
import geopandas as gpd
import pandas as pd

# Input and output
input_folder = Path("data/geojson/global_offshore_dggs")  # now a Path object
output_shp = "test/global_grid/ogim_offshore/global_offshore_grid.shp"

# Collect files
files = sorted(input_folder.glob("*.geojson")) + sorted(input_folder.glob("*.json"))

parts = []
for fp in files:
    try:
        gdf = gpd.read_file(fp)
    except Exception as e:
        print(f"Skipping {fp} due to read error: {e}")
        continue

    # Ensure CRS is EPSG 4326
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    # Keep only valid geometries
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    if not gdf.empty:
        parts.append(gdf)

if not parts:
    raise ValueError("No valid geometries found in the folder")

combined = pd.concat(parts, ignore_index=True, sort=True)
combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

# Save as Shapefile
combined.to_file(output_shp, driver="ESRI Shapefile")
print(f"Saved combined Shapefile to {output_shp}")
