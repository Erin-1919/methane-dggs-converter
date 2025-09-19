import os
import sys
import numpy as np
import rasterio

# Add parent directory to path to import the converter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from global_gfei_netcdf_to_dggs_optimize import GlobalGFEINetCDFToDGGSConverterOptimized


def test_gfei_raster_conversion():
    """
    Convert the FIRST NetCDF file in the configured folder to a GeoTIFF raster (Mg/year)
    using the converter's NetCDF->raster logic, and save it for quick visual inspection.
    """
    # Configuration
    netcdf_folder = r"E:/UCalgary_postdoc/data_source/GridInventory/2016-2020_Global_Fuel_Exploitation_Inventory_GFEI/2016_v1"
    geojson_folder = "data/geojson"
    output_folder = "test/test_rasters"

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Check if NetCDF folder exists
    if not os.path.exists(netcdf_folder):
        print(f"Error: NetCDF folder not found: {netcdf_folder}")
        return

    # Get list of NetCDF files
    netcdf_files = [f for f in os.listdir(netcdf_folder) if f.lower().endswith('.nc')]
    if not netcdf_files:
        print("Error: No NetCDF files found in the folder")
        return

    # Use the FIRST NetCDF file in the folder
    first_netcdf = sorted(netcdf_files)[0]
    netcdf_path = os.path.join(netcdf_folder, first_netcdf)

    print(f"Testing GFEI v3 raster conversion with file: {first_netcdf}")

    try:
        # Initialize converter with minimal mapping (only this folder)
        year_to_folder = {2016: netcdf_folder}
        converter = GlobalGFEINetCDFToDGGSConverterOptimized(
            year_to_folder=year_to_folder,
            geojson_folder=geojson_folder,
            output_folder="output",
        )

        # Extract variable name from filename using the converter's logic
        variable = converter.extract_variable_from_filename(first_netcdf)
        print(f"  Extracted variable: {variable}")

        # Convert to raster (returns Mg/year per pixel)
        raster = converter.convert_single_to_raster(netcdf_path, variable)

        # Validate raster keys
        required_keys = ['data', 'transform', 'crs', 'width', 'height']
        for k in required_keys:
            if k not in raster:
                raise ValueError(f"Raster dict missing key: {k}")

        # Stats
        data = raster['data']
        print("  Conversion stats (Mg/year):")
        print(f"    Shape: {data.shape}")
        print(f"    Min: {np.min(data):.6e}")
        print(f"    Max: {np.max(data):.6e}")
        print(f"    Mean: {np.mean(data):.6e}")
        print(f"    Total: {np.sum(data):.6e}")

        # Create output filename
        output_filename = f"gfei_V1_2016_{variable}.tiff"
        output_path = os.path.join(output_folder, output_filename)

        # Save as GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=raster['height'],
            width=raster['width'],
            count=1,
            dtype=data.dtype,
            crs=raster['crs'],
            transform=raster['transform'],
            nodata=0.0,
        ) as dst:
            dst.write(data, 1)

        print(f"  Saved raster to: {output_path}")

        # Verify the saved file
        with rasterio.open(output_path) as src:
            loaded = src.read(1)
            print("  Verification - Loaded data:")
            print(f"    Shape: {loaded.shape}")
            print(f"    Min: {np.min(loaded):.6e}")
            print(f"    Max: {np.max(loaded):.6e}")
            print(f"    Total: {np.sum(loaded):.6e}")

        print("\nGFEI raster conversion test completed successfully!")
        print(f"Output file saved in: {output_path}")

    except Exception as e:
        print(f"Error during GFEI v3 raster conversion test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_gfei_raster_conversion()



