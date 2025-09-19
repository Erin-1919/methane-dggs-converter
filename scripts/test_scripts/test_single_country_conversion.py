import os
import re
import time
import logging
import multiprocessing
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.transform import from_origin
import rasterio.windows
from shapely.geometry import box


def _setup_logger():
    if logging.getLogger().handlers:
        return logging.getLogger(__name__), None
    log_folder = "log"
    os.makedirs(log_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"test_single_country_conversion_{timestamp}.log"
    log_path = os.path.join(log_folder, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), log_path


def _create_temp_raster_from_netcdf(ds):
    if 'emissions' not in ds.variables:
        raise ValueError("Required variable 'emissions' not found in NetCDF file")
    emissions = ds['emissions'].values
    if emissions.ndim == 4:
        emissions = emissions[0, :, :, :]
    elif emissions.ndim == 3:
        emissions = emissions[0, :, :]
    elif emissions.ndim == 2:
        emissions = emissions
    else:
        raise ValueError(f"Unexpected emissions dimensions: {emissions.shape}")
    lat = ds['lat'].values
    lon = ds['lon'].values
    # Orient north-up
    lat = lat[::-1]
    emissions = emissions[::-1, :]
    # Derive resolution from lon/lat spacing
    if len(lon) > 1:
        dx = float(abs(lon[1] - lon[0]))
    else:
        dx = 0.1
    if len(lat) > 1:
        dy = float(abs(lat[0] - lat[1]))
    else:
        dy = 0.1
    emissions = np.nan_to_num(emissions, nan=0.0, posinf=0.0, neginf=0.0)
    emissions = np.clip(emissions, 0, None)
    half_x = dx / 2.0
    half_y = dy / 2.0
    top_left_lon = lon.min() - half_x
    top_left_lat = lat.max() + half_y
    transform = from_origin(top_left_lon, top_left_lat, dx, dy)
    temp_folder = "temp"
    os.makedirs(temp_folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_path = os.path.join(temp_folder, f"temp_raster_{ts}.tiff")
    with rasterio.open(
        temp_path,
        'w',
        driver='GTiff',
        height=emissions.shape[0],
        width=emissions.shape[1],
        count=1,
        dtype=emissions.dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=0.0,
    ) as dst:
        dst.write(emissions, 1)
    return temp_path


def _clip_raster_by_bbox(raster_path, bbox):
    minx, miny, maxx, maxy = bbox
    buffer = 0.05
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer
    with rasterio.open(raster_path) as src:
        window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
        clipped = src.read(1, window=window)
        clipped_transform = rasterio.windows.transform(window, src.transform)
        bounds = rasterio.transform.array_bounds(clipped.shape[0], clipped.shape[1], clipped_transform)
        return {
            'data': clipped,
            'transform': clipped_transform,
            'bounds': bounds,
            'crs': 'EPSG:4326',
            'width': clipped.shape[1],
            'height': clipped.shape[0]
        }


def _pixel_to_dggs_worker(args):
    grid, sindex, data, transform, rows, cols = args
    try:
        from rasterio.transform import xy
        a = float(transform.a)
        e = float(transform.e)
        px_w = abs(a)
        px_h = abs(e)
        pixel_area = px_w * px_h
        contributions = defaultdict(float)
        chunk_target_sum = 0.0
        
        for r, c in zip(rows, cols):
            value = float(data[r, c])
            if value <= 0.0:
                continue
            cx, cy = xy(transform, int(r), int(c), offset='center')
            half_x = px_w / 2.0
            half_y = px_h / 2.0
            pg = box(cx - half_x, cy - half_y, cx + half_x, cy + half_y)
            try:
                cand_idx = list(sindex.intersection(pg.bounds))
            except Exception:
                cand_idx = list(range(len(grid)))
            if not cand_idx:
                continue
            total_area = 0.0
            areas = []
            for idx in cand_idx:
                try:
                    inter = pg.intersection(grid.iloc[idx].geometry)
                    a = float(inter.area)
                except Exception:
                    a = 0.0
                areas.append(a)
                total_area += a
            # Buffered retry to mitigate edge-touching precision cases
            if total_area == 0.0 and cand_idx:
                try:
                    eps = min(px_w, px_h) * 1e-9
                    pg_eps = pg.buffer(eps)
                    total_area = 0.0
                    areas = []
                    for idx in cand_idx:
                        try:
                            inter = pg_eps.intersection(grid.iloc[idx].geometry)
                            a = float(inter.area)
                        except Exception:
                            a = 0.0
                        areas.append(a)
                        total_area += a
                except Exception:
                    pass
            if total_area > 0.0:
                for j, idx in enumerate(cand_idx):
                    if areas[j] > 0.0:
                        weight = areas[j] / total_area
                        contributions[idx] += value * weight
                chunk_target_sum += value * (total_area / pixel_area)
            else:
                share = value / float(len(cand_idx))
                for idx in cand_idx:
                    contributions[idx] += share
        
        if contributions:
            idx_array = np.fromiter(contributions.keys(), dtype=np.int64)
            val_array = np.fromiter(contributions.values(), dtype=np.float64)
        else:
            idx_array = np.empty(0, dtype=np.int64)
            val_array = np.empty(0, dtype=np.float64)
        return idx_array, val_array, chunk_target_sum
    except Exception as e:
        print(f"Error in pixel worker: {e}")
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 0.0


class SingleCountryNetCDFToDGGSConverter:
    """
    Simplified converter for single country and single NetCDF file.
    """

    def __init__(self, country_geojson_path, netcdf_path, output_folder, ipcc_code, year, chunk_size_pixels=20000):
        self.country_geojson_path = country_geojson_path
        self.netcdf_path = netcdf_path
        self.output_folder = output_folder
        self.ipcc_code = ipcc_code
        self.year = year
        self.chunk_size_pixels = chunk_size_pixels

        self.logger, self.log_path = _setup_logger()
        self.log_message(f"Processing country: {country_geojson_path}")
        self.log_message(f"Processing NetCDF: {netcdf_path}")
        self.log_message(f"IPCC Code: {ipcc_code}, Year: {year}")

        # Load country grid
        self._load_country_grid()
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

    def log_message(self, message):
        print(message)
        self.logger.info(message)

    def _load_country_grid(self):
        self.log_message(f"Loading country grid from: {self.country_geojson_path}")
        self.country_grid = gpd.read_file(self.country_geojson_path)
        required_columns = ['zoneID', 'GID']
        missing = [c for c in required_columns if c not in self.country_grid.columns]
        if missing:
            raise ValueError(f"Missing required columns in country GeoJSON: {missing}")
        self.log_message(f"Loaded country grid with {len(self.country_grid)} DGGS cells")
        
        # Create spatial index
        self.country_spatial_index = self.country_grid.sindex
        self.country_bounds = self.country_grid.geometry.union_all().bounds
        self.log_message(f"Country bounds: {self.country_bounds}")

    def process_netcdf(self):
        self.log_message("Processing NetCDF file...")
        
        try:
            # Load NetCDF and create temporary raster
            ds = xr.open_dataset(self.netcdf_path)
            temp_tiff_path = _create_temp_raster_from_netcdf(ds)
            self.log_message(f"Created temporary raster: {temp_tiff_path}")
        except Exception as e:
            self.log_message(f"Error loading NetCDF {self.netcdf_path}: {e}")
            return None

        try:
            # Clip raster to country bounds
            clipped = _clip_raster_by_bbox(temp_tiff_path, self.country_bounds)
            self.log_message(f"Clipped raster shape: {clipped['data'].shape}")
            
            # Find non-zero pixels
            data = clipped['data']
            mask = data > 0
            if not np.any(mask):
                self.log_message("No non-zero pixels found in clipped raster")
                return self._create_empty_result()
            
            rows, cols = np.where(mask)
            self.log_message(f"Found {len(rows)} non-zero pixels")
            
            # Process pixels in chunks
            n = len(rows)
            chunk = self.chunk_size_pixels
            tasks = []
            
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                tasks.append((
                    self.country_grid,
                    self.country_spatial_index,
                    data,
                    clipped['transform'],
                    rows[start:end],
                    cols[start:end]
                ))
            
            self.log_message(f"Created {len(tasks)} pixel processing tasks")
            
            # Process chunks
            aggregate_map = defaultdict(float)
            target_sum = 0.0
            
            for task in tasks:
                idx_array, val_array, chunk_target_sum = _pixel_to_dggs_worker(task)
                for idx, val in zip(idx_array, val_array):
                    aggregate_map[int(idx)] += float(val)
                target_sum += float(chunk_target_sum)
            
            self.log_message(f"Processed all pixel chunks. Target sum: {target_sum}")
            
            # Create results dataframe
            if not aggregate_map:
                self.log_message("No contributions found")
                return self._create_empty_result()
            
            results = np.zeros(len(self.country_grid))
            for idx, val in aggregate_map.items():
                if 0 <= idx < len(results):
                    results[idx] = results[idx] + val
            
            total = float(np.sum(results))
            if total > 0.0 and target_sum > 0.0:
                scale = target_sum / total
                results = results * scale
                self.log_message(f"Applied scaling factor: {scale:.6f}")
            
            # Create output dataframe
            df = self.country_grid[['zoneID']].copy().rename(columns={'zoneID': 'dggsID'})
            df['GID'] = self.country_grid['GID'].iloc[0]  # Get GID from first row
            df[self.ipcc_code] = results
            df['Year'] = self.year
            df = df[df[self.ipcc_code] > 0]
            
            self.log_message(f"Created results with {len(df)} non-zero records")
            return df
            
        finally:
            # Cleanup
            try:
                if os.path.exists(temp_tiff_path):
                    os.remove(temp_tiff_path)
                    self.log_message("Cleaned up temporary raster file")
            except Exception as e:
                self.log_message(f"Error cleaning up temp file: {e}")
            try:
                ds.close()
            except Exception:
                pass

    def _create_empty_result(self):
        """Create empty result dataframe with correct structure"""
        df = self.country_grid[['zoneID']].copy().rename(columns={'zoneID': 'dggsID'})
        df['GID'] = self.country_grid['GID'].iloc[0]
        df[self.ipcc_code] = 0.0
        df['Year'] = self.year
        df = df[df[self.ipcc_code] > 0]  # This will be empty
        return df

    def save_results(self, df):
        """Save results to CSV file"""
        if df is None or len(df) == 0:
            self.log_message("No results to save")
            return None
        
        gid = df['GID'].iloc[0]
        output_filename = f"EDGAR_DGGS_methane_emissions_{gid}_{self.ipcc_code}_{self.year}.csv"
        output_path = os.path.join(self.output_folder, output_filename)
        
        try:
            df.to_csv(output_path, index=False)
            self.log_message(f"Results saved to: {output_path}")
            self.log_message(f"File size: {os.path.getsize(output_path)} bytes")
            self.log_message(f"Records saved: {len(df)}")
            return output_path
        except Exception as e:
            self.log_message(f"Error saving results: {e}")
            return None


def main():
    # Configuration
    country_geojson_path = "data\geojson\global_countries_dggs_merge\Cabo_Verde_CPV_grid.geojson"
    netcdf_path = r"E:\UCalgary_postdoc\data_source\GridInventory\1970-2022_EDGAR_v8.0_Greenhouse_Gas_CH4_Emissions\ENF_emi_nc\v8.0_FT2022_GHG_CH4_2015_ENF_emi.nc"
    output_folder = "output"
    ipcc_code = "3A1"  # ENF sector maps to 3A1 according to EDGAR lookup
    year = 2015

    # Check if files exist
    if not os.path.exists(country_geojson_path):
        print(f"Error: Country GeoJSON file not found: {country_geojson_path}")
        return
    if not os.path.exists(netcdf_path):
        print(f"Error: NetCDF file not found: {netcdf_path}")
        return

    # Create converter and process
    converter = SingleCountryNetCDFToDGGSConverter(
        country_geojson_path,
        netcdf_path,
        output_folder,
        ipcc_code,
        year
    )
    
    try:
        # Process the NetCDF file
        results_df = converter.process_netcdf()
        
        if results_df is not None:
            # Save results
            output_path = converter.save_results(results_df)
            if output_path:
                converter.log_message(f"\nConversion completed successfully!")
                converter.log_message(f"Output file: {output_path}")
            else:
                converter.log_message("Failed to save results")
        else:
            converter.log_message("No results generated")
            
    except Exception as e:
        converter.log_message(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
