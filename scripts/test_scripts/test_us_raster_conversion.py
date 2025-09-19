import os
import sys
import numpy as np
import xarray as xr
import rasterio
from pathlib import Path

# Add parent directory to path to import the converter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from us_netcdf_to_dggs_converter import USNetCDFToDGGSConverterAggregated

def test_raster_conversion():
    """
    Test script to convert the first three variables from the first NetCDF file
    to raster format and save as GeoTIFF files.
    """
    # Configuration
    netcdf_folder = "E:/UCalgary_postdoc/data_source/GridInventory/2012-2018_U.S._Anthropogenic_Methane_Emissions"
    output_folder = "test/test_rasters"
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if NetCDF folder exists
    if not os.path.exists(netcdf_folder):
        print(f"Error: NetCDF folder not found: {netcdf_folder}")
        return
    
    # Get list of NetCDF files
    netcdf_files = [f for f in os.listdir(netcdf_folder) if f.endswith('.nc')]
    if not netcdf_files:
        print("Error: No NetCDF files found in the folder")
        return
    
    # Use the first NetCDF file
    first_netcdf = netcdf_files[0]
    netcdf_path = os.path.join(netcdf_folder, first_netcdf)
    
    print(f"Testing raster conversion with file: {first_netcdf}")
    
    try:
        # Load NetCDF data
        nc_data = xr.open_dataset(netcdf_path)
        
        # Get variables (excluding coordinates, time, and area)
        exclude_vars = ['lat', 'lon', 'time', 'grid_cell_area']
        variables = [var for var in nc_data.variables if var not in exclude_vars]
        
        print(f"Found {len(variables)} variables in the NetCDF file")
        
        # Take only the first three variables for testing
        test_variables = variables[:3]
        print(f"Testing with variables: {test_variables}")
        
        # Extract coordinates and area
        lat = nc_data['lat'].values
        lon = nc_data['lon'].values
        area = nc_data['grid_cell_area'].values if 'grid_cell_area' in nc_data.variables else None
        
        if area is None:
            raise ValueError("Required variable 'grid_cell_area' not found in NetCDF file")
        
        # Handle area variable dimensions if it has time dimension
        if area.ndim == 4:  # (time, lat, lon, other)
            area = area[0, :, :, :]  # Take first time step
        elif area.ndim == 3:  # (time, lat, lon)
            area = area[0, :, :]  # Take first time step
        elif area.ndim == 2:  # (lat, lon)
            area = area  # Already 2D
        else:
            raise ValueError(f"Unexpected area dimensions: {area.shape}")
        
        print(f"  Area variable shape: {area.shape}")
        print(f"  Latitude shape: {lat.shape}")
        print(f"  Longitude shape: {lon.shape}")
        
        # Ensure latitude is in correct order (top to bottom)
        lat = lat[::-1]
        
        # Create transform for raster
        cell_size = 0.1  # 0.1 degree resolution
        half_cell = cell_size / 2.0
        
        # Calculate the top-left corner of the raster
        top_left_lon = lon.min() - half_cell
        top_left_lat = lat.max() + half_cell
        
        transform = rasterio.transform.from_origin(top_left_lon, top_left_lat, cell_size, cell_size)
        
        # Constants for unit conversion
        AVOGADRO = 6.022e23  # molecules/mol
        M_CH4 = 16.04        # g/mol
        G_TO_MG = 1e-6       # grams to megagrams (Mg): 1 Mg = 1,000,000 g
        
        # Calculate seconds per year (assuming annual data)
        seconds_per_year = 365 * 24 * 3600
        
        # Extract year from filename
        import re
        year_match = re.search(r'(\d{4})\.nc$', first_netcdf)
        if year_match:
            year = year_match.group(1)
        else:
            year = "unknown"
        
        # Process each test variable
        for i, var_name in enumerate(test_variables):
            print(f"\nProcessing variable {i+1}: {var_name}")
            
            # Get variable data
            var_data = nc_data[var_name].values
            
            print(f"    Original data shape: {var_data.shape}")
            
            # Handle multi-dimensional data - take first time step if time dimension exists
            if var_data.ndim == 4:  # (time, lat, lon, other)
                var_data = var_data[0, :, :, :]  # Take first time step
                print(f"    After removing time dimension: {var_data.shape}")
            elif var_data.ndim == 3:  # (time, lat, lon)
                var_data = var_data[0, :, :]  # Take first time step
                print(f"    After removing time dimension: {var_data.shape}")
            elif var_data.ndim == 2:  # (lat, lon)
                var_data = var_data  # Already 2D
                print(f"    Data is already 2D: {var_data.shape}")
            else:
                raise ValueError(f"Unexpected data dimensions: {var_data.shape}")
            
            # Reverse latitude order to match raster format
            var_data = var_data[::-1, :]
            print(f"    Final data shape after latitude reversal: {var_data.shape}")
            
            # Convert flux units: molecules CH₄ cm⁻² s⁻¹ to Mg/year
            # mass_Mg = (flux * area * seconds_per_year / AVOGADRO) * M_CH4 * G_TO_MG
            
            # Ensure var_data and area have compatible shapes
            if var_data.shape != area.shape:
                print(f"    Warning: Shape mismatch - var_data: {var_data.shape}, area: {area.shape}")
                # Try to broadcast if possible
                try:
                    mass_per_pixel = (var_data * area * seconds_per_year / AVOGADRO) * M_CH4 * G_TO_MG
                except ValueError as e:
                    print(f"    Error in calculation: {e}")
                    print(f"    Skipping this variable due to shape incompatibility")
                    continue
            else:
                mass_per_pixel = (var_data * area * seconds_per_year / AVOGADRO) * M_CH4 * G_TO_MG
            
            # Handle missing data
            mass_per_pixel = np.nan_to_num(mass_per_pixel, nan=0.0)
            
            # Validate conversion results
            print(f"  Conversion stats:")
            print(f"    Min: {np.min(mass_per_pixel):.2e} Mg/year")
            print(f"    Max: {np.max(mass_per_pixel):.2e} Mg/year")
            print(f"    Mean: {np.mean(mass_per_pixel):.2e} Mg/year")
            print(f"    Total: {np.sum(mass_per_pixel):.2e} Mg/year")
            
            # Extract code prefix for filename
            if var_name.startswith('emi_ch4_'):
                remaining = var_name[8:]  # length of 'emi_ch4_'
                underscore_pos = remaining.find('_')
                if underscore_pos != -1:
                    code_prefix = remaining[:underscore_pos]
                else:
                    code_prefix = remaining
            else:
                code_prefix = var_name
            
            # Create output filename
            output_filename = f"us_{code_prefix}_{year}.tiff"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save as GeoTIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=len(lat),
                width=len(lon),
                count=1,
                dtype=mass_per_pixel.dtype,
                crs='EPSG:4326',
                transform=transform,
                nodata=0.0
            ) as dst:
                dst.write(mass_per_pixel, 1)
            
            print(f"  Saved raster to: {output_path}")
            
            # Verify the saved file
            with rasterio.open(output_path) as src:
                data = src.read(1)
                print(f"  Verification - Loaded data:")
                print(f"    Shape: {data.shape}")
                print(f"    Min: {np.min(data):.2e} Mg/year")
                print(f"    Max: {np.max(data):.2e} Mg/year")
                print(f"    Total: {np.sum(data):.2e} Mg/year")
        
        nc_data.close()
        print(f"\nRaster conversion test completed successfully!")
        print(f"Output files saved in: {output_folder}")
        
    except Exception as e:
        print(f"Error during raster conversion test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_raster_conversion()
