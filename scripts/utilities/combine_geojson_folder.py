import os
import geopandas as gpd
import pandas as pd

# === Paths ===
input_folder = r"data/geojson/global_countries_dggs_merge"
output_file = r"data/geojson/global_countries_dggs_merge.geojson"

# === Expected schema ===
expected_cols = {"NAME", "GID", "zoneID", "geometry"}

# === Collect GeoJSON files ===
geojson_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".geojson")]
if not geojson_files:
    raise FileNotFoundError(f"No GeoJSON files found in {input_folder}")

print(f"Found {len(geojson_files)} GeoJSON files to merge\n")

# === Read and validate each file ===
gdf_list = []
for i, file in enumerate(geojson_files, 1):
    file_path = os.path.join(input_folder, file)
    print(f"[{i}/{len(geojson_files)}] Reading {file_path}")
    
    gdf = gpd.read_file(file_path)
    
    # Schema check
    if not expected_cols.issubset(gdf.columns):
        raise ValueError(f" {file_path} does not match expected schema {expected_cols}. "
                         f"Found columns: {list(gdf.columns)}")
    
    # CRS alignment
    if gdf_list:
        gdf = gdf.to_crs(gdf_list[0].crs)
    
    gdf_list.append(gdf)

# === Combine all into one GeoDataFrame ===
merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)

# === Drop duplicates on zoneID ===
before = len(merged_gdf)
dupes = merged_gdf[merged_gdf.duplicated("zoneID", keep=False)]

if not dupes.empty:
    print(f"\n Found {len(dupes)} duplicate rows based on 'zoneID'")
    print(dupes[["NAME", "GID", "zoneID"]].head())
    
merged_gdf = merged_gdf.drop_duplicates(subset="zoneID", keep="first")
after = len(merged_gdf)

print(f"\nRemoved {before - after} duplicate records based on 'zoneID'")
print(f"Final record count: {after}")

# === Save merged file ===
merged_gdf.to_file(output_file, driver="GeoJSON")
print(f"\n Combined GeoJSON saved to {output_file}")
