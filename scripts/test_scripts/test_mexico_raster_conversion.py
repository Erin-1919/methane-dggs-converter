import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
import warnings

# Filter out warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def _extract_code_prefix(variable_name):
    """
    Extract the code prefix from variable names like:
    - emis_ch4_4D1_treatment -> 4D1
    - emis_ch4_4D2_no_treatment -> 4D2
    - emis_ch4_3A1a -> 3A1a
    - emis_ch4_1A3a,c,d -> 1A3a-1A3c-1A3d
    
    Args:
        variable_name (str): Full variable name from NetCDF
        
    Returns:
        str: Extracted code prefix with commas converted to hyphens
    """
    # Split by underscores
    parts = variable_name.split('_')
    
    # Look for the pattern: emis_ch4_CODE_otherinfo
    # The code is typically the 3rd part (index 2) after emis_ch4
    if len(parts) >= 3 and parts[0] == 'emis' and parts[1] == 'ch4':
        code_prefix = parts[2]
        
        # Convert commas to expanded hyphen-separated format for CSV compatibility and readability
        # This converts "1A3a,c,d" to "1A3a-1A3c-1A3d"
        if ',' in code_prefix:
            # Split by comma
            code_parts = code_prefix.split(',')
            # Ensure each part is properly formatted (remove any extra spaces)
            code_parts = [part.strip() for part in code_parts]
            
            # Extract the base prefix (e.g., "1A3" from "1A3a,c,d")
            # Find the common prefix by looking for the longest common start
            if len(code_parts) > 1:
                # Find the common base prefix
                base_prefix = ""
                first_part = code_parts[0]
                for i in range(len(first_part)):
                    if all(first_part[i] == part[i] for part in code_parts if len(part) > i):
                        base_prefix += first_part[i]
                    else:
                        break
                
                # If we found a common base, expand each part
                if base_prefix:
                    expanded_parts = []
                    for part in code_parts:
                        if part.startswith(base_prefix):
                            expanded_parts.append(part)
                        else:
                            expanded_parts.append(f"{base_prefix}{part}")
                    code_prefix = '-'.join(expanded_parts)
                else:
                    # No common base found, just join with hyphens
                    code_prefix = '-'.join(code_parts)
            else:
                # Single part, no expansion needed
                code_prefix = code_parts[0]
            
            print(f"      Converted comma-separated code '{parts[2]}' to '{code_prefix}'")
        
        print(f"      Extracted code '{code_prefix}' from '{variable_name}'")
        print(f"      DEBUG: Returning code_prefix: '{code_prefix}'")
        return code_prefix
    
    # Fallback: if pattern doesn't match, return the original name
    print(f"      No pattern match for '{variable_name}', using original name")
    return variable_name


def aggregate_variables_by_code(nc_data):
    """
    Aggregate variables with the same code prefix at the xarray level.
    
    Args:
        nc_data (xarray.Dataset): NetCDF dataset
        
    Returns:
        dict: Dictionary with aggregated data for each code prefix
    """
    print("Aggregating variables by code prefix...")
    
    # Get variables (excluding coordinates and area)
    exclude_vars = ['lat', 'lon', 'area']
    variables = [var for var in nc_data.variables if var not in exclude_vars]
    
    # Check if there's only one variable available
    if len(variables) == 1:
        # Only one variable - keep it regardless of 'total' in name
        var = variables[0]
        code_prefix = _extract_code_prefix(var)
        print(f"  Single variable found: {var}")
        print(f"  Extracted code prefix: {code_prefix}")
        
        return {code_prefix: nc_data[var].values}
    
    # Multiple variables - filter out those containing 'total' and 'other'
    variables = [var for var in variables if 'total' not in var.lower() and 'other' not in var.lower()]
    print(f"  Filtered variables (excluded 'total' and 'other'): {variables}")
    
    # Group variables by code prefix
    code_groups = {}
    for var in variables:
        # Extract code prefix using the intelligent method
        code_prefix = _extract_code_prefix(var)
        
        if code_prefix not in code_groups:
            code_groups[code_prefix] = []
        code_groups[code_prefix].append(var)
    
    print(f"  Found {len(code_groups)} unique code prefixes:")
    for code, vars_list in code_groups.items():
        print(f"    {code}: {vars_list}")
    
    # Aggregate variables for each code prefix
    aggregated_data = {}
    for code_prefix, var_list in code_groups.items():
        print(f"  Aggregating {code_prefix}: {var_list}")
        print(f"    DEBUG: code_prefix type: {type(code_prefix)}, value: '{code_prefix}'")
        
        if len(var_list) == 1:
            # Single variable - just use it directly
            aggregated_data[code_prefix] = nc_data[var_list[0]].values
            print(f"    Single variable: {var_list[0]}")
        else:
            # Multiple variables - sum them up
            print(f"    Summing {len(var_list)} variables...")
            # Start with the first variable
            aggregated_array = nc_data[var_list[0]].values
            
            # Add the rest
            for var in var_list[1:]:
                aggregated_array = aggregated_array + nc_data[var].values
            
            aggregated_data[code_prefix] = aggregated_array
            print(f"    Aggregated into single array: {aggregated_array.shape}")
    
    print(f"  DEBUG: Final aggregated_data keys: {list(aggregated_data.keys())}")
    return aggregated_data

def convert_netcdf_to_raster(netcdf_path, output_folder):
    """
    Convert a NetCDF file to raster format with pre-aggregated variables.
    
    Args:
        netcdf_path (str): Path to NetCDF file
        output_folder (str): Output folder for raster files
        
    Returns:
        dict: Dictionary containing raster data for each aggregated code prefix
    """
    print(f"Processing NetCDF: {os.path.basename(netcdf_path)}")
    
    # Load NetCDF data
    nc_data = xr.open_dataset(netcdf_path)
    
    # Aggregate variables by code prefix
    aggregated_data = aggregate_variables_by_code(nc_data)
    
    # Extract coordinates
    lat = nc_data['lat'].values
    lon = nc_data['lon'].values
    
    print(f"  Original lat shape: {lat.shape}, lon shape: {lon.shape}")
    print(f"  Lat range: {lat.min():.4f} to {lat.max():.4f}")
    print(f"  Lon range: {lon.min():.4f} to {lon.max():.4f}")
    
    # Ensure latitude is in correct order (top to bottom)
    lat = lat[::-1]
    
    # Create transform for raster
    # Since lat/lon are pixel centers, we need to shift by half a pixel
    cell_size = 0.1  # 0.1 degree resolution
    half_cell = cell_size / 2.0
    
    # Calculate the top-left corner of the raster
    # lon.min() and lat.max() are pixel centers, so subtract half a pixel
    top_left_lon = lon.min() - half_cell
    top_left_lat = lat.max() + half_cell
    
    transform = from_origin(top_left_lon, top_left_lat, cell_size, cell_size)
    
    print(f"  Raster transform: {transform}")
    print(f"  Raster dimensions: {len(lon)} x {len(lat)}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load area data from Canada file for proper unit conversion
    area_file_path = "E:/UCalgary_postdoc/data_source/GridInventory/2018_Canada_Anthropogenic_Methane_Emissions/can_emis_coal_2018.nc"
    area_data = None
    if os.path.exists(area_file_path):
        try:
            canada_nc = xr.open_dataset(area_file_path)
            area_data = canada_nc['area'].values
            canada_nc.close()
            print(f"  Loaded area data with shape: {area_data.shape}")
        except Exception as e:
            print(f"  Warning: Could not load area data: {e}")
            print("  Will use uniform area assumption")
    else:
        print(f"  Warning: Area file not found: {area_file_path}")
        print("  Will use uniform area assumption")
    
    raster_data = {}
    
    for code_prefix, aggregated_array in aggregated_data.items():
        print(f"  Processing aggregated code: {code_prefix}")
        print(f"    DEBUG: This code_prefix came from aggregated_data.keys(): {list(aggregated_data.keys())}")
        
        # Reverse latitude order to match raster format
        aggregated_array = aggregated_array[::-1, :]
        
        # Convert units: Mg/year/km² to Mg/year/m²
        # 1 km² = 1,000,000 m²
        # So Mg/year/km² × (1 km² / 1,000,000 m²) = Mg/year/m² × 10⁻⁶
        aggregated_array_mg_per_m2 = aggregated_array * 1e-6
        
        # Calculate total emissions per pixel using area data to get Mg/year
        if area_data is not None:
            # area is in m², so multiply to get total emissions per pixel in Mg/year
            total_emission_per_pixel = aggregated_array_mg_per_m2 * area_data
            print(f"    Applied area multiplication to convert to Mg/year")
        else:
            # If no area data, assume uniform area (this will not give correct Mg/year values)
            total_emission_per_pixel = aggregated_array_mg_per_m2
            print(f"    Warning: No area data available, output will be in Mg/year/m², not Mg/year")
        
        # Handle missing data
        total_emission_per_pixel = np.nan_to_num(total_emission_per_pixel, nan=0.0)
        
        # Save as GeoTIFF for visualization
        output_filename = f"mexico_{code_prefix}_raster.tif"
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"    Saving raster to: {output_path}")
        print(f"    Data shape: {total_emission_per_pixel.shape}")
        print(f"    Data range: {total_emission_per_pixel.min():.6f} to {total_emission_per_pixel.max():.6f}")
        print(f"    Non-zero pixels: {np.count_nonzero(total_emission_per_pixel)}")
        print(f"    Units: Mg/year (Megagrams per year)")
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=total_emission_per_pixel.shape[0],
            width=total_emission_per_pixel.shape[1],
            count=1,
            dtype=total_emission_per_pixel.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(total_emission_per_pixel, 1)
        
        # Store raster data for return
        raster_data[code_prefix] = {
            'data': total_emission_per_pixel,
            'transform': transform,
            'crs': 'EPSG:4326',
            'width': len(lon),
            'height': len(lat),
            'lat': lat,
            'lon': lon,
            'output_path': output_path
        }
    
    nc_data.close()
    return raster_data
    
def main():
    """
    Main function to test raster conversion.
    """
    # Configuration
    netcdf_folder = "E:/UCalgary_postdoc/data_source/GridInventory/2015_Mexico_Anthropogenic_Methane_Emissions"
    output_folder = "test/test_rasters"
    
    # Check if input folder exists
    if not os.path.exists(netcdf_folder):
        print(f"Error: NetCDF folder not found: {netcdf_folder}")
        return
    
    # Get list of NetCDF files
    netcdf_files = [f for f in os.listdir(netcdf_folder) if f.endswith('.nc')]
    
    if not netcdf_files:
        print(f"No NetCDF files found in: {netcdf_folder}")
        return
    
    # Use the first NetCDF file for testing
    first_netcdf = netcdf_files[0]
    netcdf_path = os.path.join(netcdf_folder, first_netcdf)
    
    print(f"Testing raster conversion with: {first_netcdf}")
    print(f"Input path: {netcdf_path}")
    print(f"Output folder: {output_folder}")
    print("=" * 60)
    
    try:
        # Convert NetCDF to raster
        raster_data = convert_netcdf_to_raster(netcdf_path, output_folder)
        
        print("\n" + "=" * 60)
        print("RASTER CONVERSION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Summary of what was created
        print(f"\nCreated {len(raster_data)} raster files:")
        for code_prefix, data in raster_data.items():
            print(f"  {code_prefix}: {data['output_path']}")
            print(f"    Dimensions: {data['width']} x {data['height']}")
            print(f"    Data range: {data['data'].min():.6f} to {data['data'].max():.6f}")
            print(f"    Non-zero pixels: {np.count_nonzero(data['data'])}")
        
        print(f"\nRaster files saved in: {os.path.abspath(output_folder)}")
        print("You can now visualize these files in GIS software like QGIS or ArcGIS")
        
    except Exception as e:
        print(f"Error during raster conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
