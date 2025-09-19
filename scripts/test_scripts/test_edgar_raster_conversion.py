import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
import xarray as xr


def test_edgar_raster_conversion_specific():
    """
    Convert the specified EDGAR NetCDF to a GeoTIFF raster (Mg/year) and save
    it to test/test_rasters for visual inspection.
    """
    netcdf_path = r"E:\UCalgary_postdoc\data_source\GridInventory\1970-2022_EDGAR_v8.0_Greenhouse_Gas_CH4_Emissions\ENF_emi_nc\v8.v8.0_FT2022_GHG_CH4_2015_ENF_emi.nc"
    output_folder = os.path.join("test", "test_rasters")
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(netcdf_path):
        print(f"Error: NetCDF not found: {netcdf_path}")
        return

    print(f"Testing EDGAR raster conversion with file: {os.path.basename(netcdf_path)}")

    try:
        ds = xr.open_dataset(netcdf_path)

        if 'emissions' not in ds.variables:
            raise ValueError("Variable 'emissions' not found in NetCDF")

        data = ds['emissions'].values
        # Collapse leading dimensions to first slice if present
        if data.ndim == 4:
            data = data[0, :, :, :]
        elif data.ndim == 3:
            data = data[0, :, :]
        elif data.ndim == 2:
            data = data
        else:
            raise ValueError(f"Unexpected emissions dimensions: {data.shape}")

        lat = ds['lat'].values
        lon = ds['lon'].values

        # Orient to north-up and compute per-pixel totals (already Mg/year per pixel in EDGAR)
        lat = lat[::-1]
        data = data[::-1, :]

        # Derive resolution from lon/lat spacing
        if len(lon) > 1:
            dx = float(abs(lon[1] - lon[0]))
        else:
            dx = 0.1
        if len(lat) > 1:
            dy = float(abs(lat[0] - lat[1]))
        else:
            dy = 0.1

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.clip(data, 0, None)

        # Build transform using cell centers convention
        half_x = dx / 2.0
        half_y = dy / 2.0
        top_left_lon = lon.min() - half_x
        top_left_lat = lat.max() + half_y
        transform = from_origin(top_left_lon, top_left_lat, dx, dy)

        # Save GeoTIFF
        out_name = "edgar_ENF_2015.tiff"
        output_path = os.path.join(output_folder, out_name)
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=transform,
            nodata=0.0,
        ) as dst:
            dst.write(data, 1)

        print(f"  Saved raster to: {output_path}")

        # Quick verification
        with rasterio.open(output_path) as src:
            loaded = src.read(1)
            print("  Verification - Loaded data:")
            print(f"    Shape: {loaded.shape}")
            print(f"    Min: {np.min(loaded):.6e}")
            print(f"    Max: {np.max(loaded):.6e}")
            print(f"    Total: {np.sum(loaded):.6e}")

        print("\nEDGAR raster conversion test completed successfully!")
        print(f"Output file saved in: {output_path}")

    except Exception as e:
        print(f"Error during EDGAR raster conversion test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            ds.close()
        except Exception:
            pass


if __name__ == "__main__":
    test_edgar_raster_conversion_specific()


