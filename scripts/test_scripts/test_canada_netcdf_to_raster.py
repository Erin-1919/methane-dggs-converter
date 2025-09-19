import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
import warnings

# Filter out warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def test_netcdf_to_raster():
    """
    Test converting NetCDF to raster and save as physical file for visualization.
    """
    print("=== Testing NetCDF to Raster Conversion ===")
    
    # Configuration
    netcdf_folder = "E:/UCalgary_postdoc/data_source/GridInventory/2018_Canada_Anthropogenic_Methane_Emissions"
    output_folder = "test/test_rasters"
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of NetCDF files
    netcdf_files = [f for f in os.listdir(netcdf_folder) if f.endswith('.nc')]
    
    if not netcdf_files:
        print("No NetCDF files found in the specified folder.")
        return
    
    # Use the first NetCDF file for testing
    test_netcdf = netcdf_files[0]
    test_path = os.path.join(netcdf_folder, test_netcdf)
    
    print(f"Testing with NetCDF file: {test_netcdf}")
    
    try:
        # Load NetCDF data
        print("Loading NetCDF data...")
        nc_data = xr.open_dataset(test_path)
        
        # Get variables (excluding coordinates and area)
        exclude_vars = ['lat', 'lon', 'area']
        variables = [var for var in nc_data.variables if var not in exclude_vars]
        
        print(f"Found {len(variables)} emission variables:")
        for var in variables:
            print(f"  - {var}")
        
        # Extract coordinates and area
        lat = nc_data['lat'].values
        lon = nc_data['lon'].values
        area = nc_data['area'].values if 'area' in nc_data.variables else None
        
        print(f"\nGrid dimensions: {len(lon)} x {len(lat)}")
        print(f"Longitude range: {lon.min():.3f} to {lon.max():.3f}")
        print(f"Latitude range: {lat.min():.3f} to {lat.max():.3f}")
        
        if area is not None:
            print(f"Area range: {area.min():.1f} to {area.max():.1f} m²")
        
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
        
        print(f"\nRaster transform:")
        print(f"  Top-left corner: ({top_left_lon:.6f}, {top_left_lat:.6f})")
        print(f"  Cell size: {cell_size} degrees")
        print(f"  Width: {len(lon)}, Height: {len(lat)}")
        
        # Process each variable and save as raster
        for i, variable in enumerate(variables):
            print(f"\n--- Processing variable {i+1}/{len(variables)}: {variable} ---")
            
            # Extract variable data
            variable_data = nc_data[variable].values
            
            # Reverse latitude order to match raster format
            variable_data = variable_data[::-1, :]
            
            print(f"  Original data range: {variable_data.min():.6f} to {variable_data.max():.6f}")
            print(f"  Original data shape: {variable_data.shape}")
            
            # Convert units: Mg/year/km² to Mg/year/m²
            # 1 km² = 1,000,000 m²
            # So Mg/year/km² × (1 km² / 1,000,000 m²) = Mg/year/m² × 10⁻⁶
            variable_data_mg_per_m2 = variable_data * 1e-6
            
            print(f"  After km²→m² conversion: {variable_data_mg_per_m2.min():.2e} to {variable_data_mg_per_m2.max():.2e} Mg/year/m²")
            
            # Calculate total emissions per pixel if area is available
            if area is not None:
                # area is in m², so multiply to get total emissions per pixel in Mg/year
                total_emission_per_pixel = variable_data_mg_per_m2 * area
                print(f"  After area multiplication: {total_emission_per_pixel.min():.2e} to {total_emission_per_pixel.max():.2e} Mg/year")
            else:
                print("  Warning: No area data found, using original values")
                total_emission_per_pixel = variable_data_mg_per_m2
            
            # Keep in Mg/year (Megagrams per year) - no conversion to kilotons
            # total_emission_per_pixel is now in Mg/year
            
            print(f"  Final values (Mg/year): {total_emission_per_pixel.min():.2e} to {total_emission_per_pixel.max():.2e}")
            
            # Handle missing data
            total_emission_per_pixel = np.nan_to_num(total_emission_per_pixel, nan=0.0)
            
            # Save as GeoTIFF
            output_filename = f"test_{os.path.splitext(test_netcdf)[0]}_{variable.replace(' ', '_').replace('(', '').replace(')', '')}.tif"
            output_path = os.path.join(output_folder, output_filename)
            
            print(f"  Saving raster to: {output_path}")
            
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                count=1,
                dtype='float32',
                width=len(lon),
                height=len(lat),
                crs='EPSG:4326',
                transform=transform,
                nodata=0
            ) as dst:
                dst.write(total_emission_per_pixel, 1)
                
                # Add metadata
                dst.update_tags(
                    title=f"Methane emissions: {variable}",
                    units="Megagrams per year",
                    source_netcdf=test_netcdf,
                    original_units="Mg/year/km²",
                    conversion_notes="Converted from Mg/year/km² to Mg/year/m²"
                )
            
            print(f"  Raster saved successfully!")
            
            # Create a simple visualization
            create_visualization(total_emission_per_pixel, variable, output_path)
        
        nc_data.close()
        
        print(f"\n=== Test Completed Successfully! ===")
        print(f"Raster files saved in: {output_folder}")
        print(f"Total files created: {len(variables)}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

def create_visualization(data, variable_name, raster_path):
    """
    Create a simple visualization of the raster data.
    """
    try:
        # Create output path for visualization
        vis_path = raster_path.replace('.tif', '_visualization.png')
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Create the main plot
        plt.subplot(2, 2, 1)
        im1 = plt.imshow(data, cmap='viridis', aspect='auto')
        plt.colorbar(im1, label='Mg/year')
        plt.title(f'{variable_name}\nFull Dataset')
        plt.xlabel('Longitude index')
        plt.ylabel('Latitude index')
        
        # Create histogram
        plt.subplot(2, 2, 2)
        non_zero_data = data[data > 0]
        if len(non_zero_data) > 0:
            plt.hist(non_zero_data, bins=50, alpha=0.7, color='blue')
            plt.xlabel('Mg/year')
            plt.ylabel('Frequency')
            plt.title('Distribution (Non-zero values)')
            plt.yscale('log')
        else:
            plt.text(0.5, 0.5, 'No non-zero values', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Distribution (No non-zero values)')
        
        # Create zoomed view of non-zero region
        plt.subplot(2, 2, 3)
        if len(non_zero_data) > 0:
            # Find bounds of non-zero region
            non_zero_mask = data > 0
            if np.any(non_zero_mask):
                rows = np.where(np.any(non_zero_mask, axis=1))[0]
                cols = np.where(np.any(non_zero_mask, axis=0))[0]
                
                if len(rows) > 0 and len(cols) > 0:
                    row_start, row_end = rows[0], rows[-1]
                    col_start, col_end = cols[0], cols[-1]
                    
                    # Add some padding
                    row_start = max(0, row_start - 10)
                    row_end = min(data.shape[0], row_end + 10)
                    col_start = max(0, col_start - 10)
                    col_end = min(data.shape[1], col_end + 10)
                    
                    zoomed_data = data[row_start:row_end, col_start:col_end]
                    im3 = plt.imshow(zoomed_data, cmap='viridis', aspect='auto')
                    plt.colorbar(im3, label='Mg/year')
                    plt.title(f'Zoomed View\nRows {row_start}-{row_end}, Cols {col_start}-{col_end}')
                    plt.xlabel('Longitude index')
                    plt.ylabel('Latitude index')
                else:
                    plt.text(0.5, 0.5, 'No non-zero regions found', ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('Zoomed View')
            else:
                plt.text(0.5, 0.5, 'No non-zero regions found', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Zoomed View')
        else:
            plt.text(0.5, 0.5, 'No non-zero values', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Zoomed View')
        
        # Statistics
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        stats_text = f"""Statistics for {variable_name}:
        
Total pixels: {data.size:,}
Non-zero pixels: {len(data[data > 0]):,}
Zero pixels: {len(data[data == 0]):,}

Value range:
Min: {data.min():.2e}
Max: {data.max():.2e}
Mean: {data.mean():.2e}
Median: {np.median(data):.2e}

Non-zero values:
Min: {data[data > 0].min() if len(data[data > 0]) > 0 else 'N/A'}
Max: {data[data > 0].max() if len(data[data > 0]) > 0 else 'N/A'}
Mean: {data[data > 0].mean() if len(data[data > 0]) > 0 else 'N/A'}

Units: Mg/year (Megagrams per year)
Source: {os.path.basename(raster_path)}"""
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Visualization saved to: {vis_path}")
        
    except Exception as e:
        print(f"    Warning: Could not create visualization: {e}")

def test_single_variable():
    """
    Test with a single variable for quick verification.
    """
    print("=== Testing Single Variable Conversion ===")
    
    # Configuration
    netcdf_folder = "E:/UCalgary_postdoc/data_source/GridInventory/2018_Canada_Anthropogenic_Methane_Emissions"
    output_folder = "test/test_rasters"
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of NetCDF files
    netcdf_files = [f for f in os.listdir(netcdf_folder) if f.endswith('.nc')]
    
    if not netcdf_files:
        print("No NetCDF files found in the specified folder.")
        return
    
    # Use the first NetCDF file
    test_netcdf = netcdf_files[0]
    test_path = os.path.join(netcdf_folder, test_netcdf)
    
    print(f"Testing with NetCDF file: {test_netcdf}")
    
    try:
        # Load NetCDF data
        nc_data = xr.open_dataset(test_path)
        
        # Get first variable only
        exclude_vars = ['lat', 'lon', 'area']
        variables = [var for var in nc_data.variables if var not in exclude_vars]
        
        if not variables:
            print("No emission variables found.")
            return
        
        variable = variables[0]  # Use first variable only
        print(f"Testing with variable: {variable}")
        
        # Extract data and convert
        variable_data = nc_data[variable].values
        lat = nc_data['lat'].values
        lon = nc_data['lon'].values
        area = nc_data['area'].values if 'area' in nc_data.variables else None
        
        # Process single variable
        lat = lat[::-1]
        variable_data = variable_data[::-1, :]
        
        # Unit conversion
        variable_data_mg_per_m2 = variable_data * 1e-6
        
        if area is not None:
            total_emission_per_pixel = variable_data_mg_per_m2 * area
        else:
            total_emission_per_pixel = variable_data_mg_per_m2
        
        # Keep in Mg/year (Megagrams per year) - no conversion to kilotons
        # total_emission_per_pixel is now in Mg/year
        total_emission_per_pixel = np.nan_to_num(total_emission_per_pixel, nan=0.0)
        
        # Create transform
        cell_size = 0.1
        half_cell = cell_size / 2.0
        top_left_lon = lon.min() - half_cell
        top_left_lat = lat.max() + half_cell
        transform = from_origin(top_left_lon, top_left_lat, cell_size, cell_size)
        
        # Save raster
        output_filename = f"single_test_{variable.replace(' ', '_').replace('(', '').replace(')', '')}.tif"
        output_path = os.path.join(output_folder, output_filename)
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            count=1,
            dtype='float32',
            width=len(lon),
            height=len(lat),
            crs='EPSG:4326',
            transform=transform,
            nodata=0
        ) as dst:
            dst.write(total_emission_per_pixel, 1)
            dst.update_tags(
                title=f"Single variable test: {variable}",
                units="Megagrams per year",
                source_netcdf=test_netcdf
            )
        
        print(f"Single variable raster saved to: {output_path}")
        print(f"Data range: {total_emission_per_pixel.min():.2e} to {total_emission_per_pixel.max():.2e} Mg/year")
        
        nc_data.close()
        
    except Exception as e:
        print(f"Error during single variable test: {e}")
        import traceback
        traceback.print_exc()

def test_specific_netcdf():
    """
    Test with a specific NetCDF file: can_emis_oil_gas_vent_flare_2018.nc
    """
    print("=== Testing Specific NetCDF: can_emis_oil_gas_vent_flare_2018.nc ===")
    
    # Configuration
    netcdf_folder = "E:/UCalgary_postdoc/data_source/GridInventory/2018_Canada_Anthropogenic_Methane_Emissions"
    output_folder = "test_rasters"
    specific_file = "can_emis_oil_gas_vent_flare_2018.nc"
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if specific file exists
    test_path = os.path.join(netcdf_folder, specific_file)
    if not os.path.exists(test_path):
        print(f"Error: Specific NetCDF file not found: {test_path}")
        print("Available files:")
        netcdf_files = [f for f in os.listdir(netcdf_folder) if f.endswith('.nc')]
        for f in netcdf_files:
            print(f"  - {f}")
        return
    
    print(f"Testing with specific NetCDF file: {specific_file}")
    
    try:
        # Load NetCDF data
        print("Loading NetCDF data...")
        nc_data = xr.open_dataset(test_path)
        
        # Get variables (excluding coordinates and area)
        exclude_vars = ['lat', 'lon', 'area']
        variables = [var for var in nc_data.variables if var not in exclude_vars]
        
        print(f"Found {len(variables)} emission variables:")
        for var in variables:
            print(f"  - {var}")
        
        # Extract coordinates and area
        lat = nc_data['lat'].values
        lon = nc_data['lon'].values
        area = nc_data['area'].values if 'area' in nc_data.variables else None
        
        print(f"\nGrid dimensions: {len(lon)} x {len(lat)}")
        print(f"Longitude range: {lon.min():.3f} to {lon.max():.3f}")
        print(f"Latitude range: {lat.min():.3f} to {lat.max():.3f}")
        
        if area is not None:
            print(f"Area range: {area.min():.1f} to {area.max():.1f} m²")
        
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
        
        print(f"\nRaster transform:")
        print(f"  Top-left corner: ({top_left_lon:.6f}, {top_left_lat:.6f})")
        print(f"  Cell size: {cell_size} degrees")
        print(f"  Width: {len(lon)}, Height: {len(lat)}")
        
        # Process each variable and save as raster
        for i, variable in enumerate(variables):
            print(f"\n--- Processing variable {i+1}/{len(variables)}: {variable} ---")
            
            # Extract variable data
            variable_data = nc_data[variable].values
            
            # Reverse latitude order to match raster format
            variable_data = variable_data[::-1, :]
            
            print(f"  Original data range: {variable_data.min():.6f} to {variable_data.max():.6f}")
            print(f"  Original data shape: {variable_data.shape}")
            
            # Convert units: Mg/year/km² to Mg/year/m²
            # 1 km² = 1,000,000 m²
            # So Mg/year/km² × (1 km² / 1,000,000 m²) = Mg/year/m² × 10⁻⁶
            variable_data_mg_per_m2 = variable_data * 1e-6
            
            print(f"  After km²→m² conversion: {variable_data_mg_per_m2.min():.2e} to {variable_data_mg_per_m2.max():.2e} Mg/year/m²")
            
            # Calculate total emissions per pixel if area is available
            if area is not None:
                # area is in m², so multiply to get total emissions per pixel in Mg/year
                total_emission_per_pixel = variable_data_mg_per_m2 * area
                print(f"  After area multiplication: {total_emission_per_pixel.min():.2e} to {total_emission_per_pixel.max():.2e} Mg/year")
            else:
                print("  Warning: No area data found, using original values")
                total_emission_per_pixel = variable_data_mg_per_m2
            
            # Keep in Mg/year (Megagrams per year) - no conversion to kilotons
            # total_emission_per_pixel is now in Mg/year
            total_emission_per_pixel = np.nan_to_num(total_emission_per_pixel, nan=0.0)
            
            # Save as GeoTIFF
            output_filename = f"oil_gas_vent_flare_{variable.replace(' ', '_').replace('(', '').replace(')', '')}.tif"
            output_path = os.path.join(output_folder, output_filename)
            
            print(f"  Saving raster to: {output_path}")
            
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                count=1,
                dtype='float32',
                width=len(lon),
                height=len(lat),
                crs='EPSG:4326',
                transform=transform,
                nodata=0
            ) as dst:
                dst.write(total_emission_per_pixel, 1)
                
                # Add metadata
                dst.update_tags(
                    title=f"Oil & Gas Vent/Flare Methane Emissions: {variable}",
                    units="Megagrams per year",
                    source_netcdf=specific_file,
                    original_units="Mg/year/km²",
                    conversion_notes="Converted from Mg/year/km² to Mg/year/m²",
                    sector="Oil and Gas Vent/Flare"
                )
            
            print(f"  Raster saved successfully!")
            
            # Create a simple visualization
            create_visualization(total_emission_per_pixel, variable, output_path)
        
        nc_data.close()
        
        print(f"\n=== Test Completed Successfully! ===")
        print(f"Raster files saved in: {output_folder}")
        print(f"Total files created: {len(variables)}")
        print(f"All files prefixed with 'oil_gas_vent_flare_'")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Choose test option:")
    print("1. Test full NetCDF to raster conversion (all variables)")
    print("2. Test single variable conversion (quick test)")
    print("3. Test specific NetCDF: can_emis_oil_gas_vent_flare_2018.nc")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        test_netcdf_to_raster()
    elif choice == "2":
        test_single_variable()
    elif choice == "3":
        test_specific_netcdf()
    else:
        print("Invalid choice. Running specific NetCDF test...")
        test_specific_netcdf()
