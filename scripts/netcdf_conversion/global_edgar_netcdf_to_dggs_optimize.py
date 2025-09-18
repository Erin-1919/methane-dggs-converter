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


# Module-level cache for worker processes
_WORKER_COUNTRY_CACHE = {}


def _setup_logger():
    if logging.getLogger().handlers:
        return logging.getLogger(__name__), None
    log_folder = "log"
    os.makedirs(log_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"global_edgar_netcdf_to_dggs_optimize_{timestamp}.log"
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


def _load_country_grid_cached(gid, grid_pickle_path):
    cached = _WORKER_COUNTRY_CACHE.get(gid)
    if cached is not None:
        return cached['grid'], cached['sindex']
    gdf = pd.read_pickle(grid_pickle_path)
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')
    sindex = gdf.sindex
    _WORKER_COUNTRY_CACHE[gid] = {'grid': gdf, 'sindex': sindex}
    return gdf, sindex


def _pixel_chunk_worker(args):
    gid, grid_pickle_path, clipped_npy_path, bounds, transform, rows, cols, ipcc_code, year, sector_folder = args
    try:
        grid, sindex = _load_country_grid_cached(gid, grid_pickle_path)
        data = np.load(clipped_npy_path, mmap_mode='r')
        # Compute pixel geometry using provided affine transform
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
        return gid, idx_array, val_array, chunk_target_sum, ipcc_code, year, sector_folder
    except Exception:
        return gid, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 0.0, ipcc_code, year, sector_folder


class GlobalEDGARNetCDFToDGGSConverterOptimized:
    """
    Optimized EDGAR NetCDF -> DGGS converter with dynamic pixel-chunk parallelism.
    Preserves original output structure and resume behavior.
    """

    def __init__(self, edgar_folder, geojson_folder, output_folder, start_year=1970, end_year=2022, max_processes=8, chunk_size_pixels=20000):
        self.edgar_folder = edgar_folder
        self.geojson_folder = geojson_folder
        self.output_folder = output_folder
        self.start_year = start_year
        self.end_year = end_year
        self.max_processes = max_processes
        self.chunk_size_pixels = chunk_size_pixels

        self.logger, self.log_path = _setup_logger()
        self.log_message(f"Processing year range: {self.start_year} to {self.end_year}")

        self.log_message("Loading EDGAR sector -> IPCC2006 lookup table...")
        self._load_edgar_lookup()

        self.log_message("Loading merged global country DGGS grid...")
        self._load_merged_country_grids()

        os.makedirs(output_folder, exist_ok=True)
        test_csv_folder = os.path.join(os.getcwd(), "test", "test_EDGAR_csv")
        os.makedirs(test_csv_folder, exist_ok=True)
        self.log_message(f"Created test_EDGAR_csv folder: {test_csv_folder}")

        self.log_message("Creating spatial index for merged grid...")
        self._create_merged_spatial_index()

        # Prepare per-country cached grid files
        self._prepare_country_grid_cache()

        # Discover sector folders
        self.sector_folders = self._get_sector_folders()
        self.log_message(f"Found {len(self.sector_folders)} sector folders")

    def log_message(self, message):
        print(message)
        self.logger.info(message)

    def _load_edgar_lookup(self):
        lookup_path = "data/lookup/edgar_ipcc_sector_mapping.csv"
        if not os.path.exists(lookup_path):
            raise FileNotFoundError(f"EDGAR lookup file not found: {lookup_path}")
        self.edgar_lookup = pd.read_csv(lookup_path)
        self.sector_to_ipcc = dict(zip(self.edgar_lookup['EDGAR Sector'], self.edgar_lookup['IPCC 2006 codes']))
        self.log_message(f"Loaded EDGAR lookup with {len(self.sector_to_ipcc)} sector mappings")

    def _load_merged_country_grids(self):
        merged_geojson_path = os.path.join(self.geojson_folder, "global_countries_dggs_merge.geojson")
        if not os.path.exists(merged_geojson_path):
            raise FileNotFoundError(f"Merged GeoJSON file not found: {merged_geojson_path}")
        self.log_message(f"Loading merged GeoJSON: {merged_geojson_path}")
        self.merged_grid = gpd.read_file(merged_geojson_path)
        required_columns = ['zoneID', 'GID']
        missing = [c for c in required_columns if c not in self.merged_grid.columns]
        if missing:
            raise ValueError(f"Missing required columns in merged GeoJSON: {missing}")
        self.log_message(f"Loaded merged grid with {len(self.merged_grid)} total DGGS cells")
        self.country_grids = {}
        self.country_geometries = {}
        for gid in self.merged_grid['GID'].unique():
            sub = self.merged_grid[self.merged_grid['GID'] == gid].copy().reset_index(drop=True)
            self.country_grids[gid] = sub
            self.country_geometries[gid] = sub.geometry.union_all()
            self.log_message(f"  Prepared {gid}: {len(sub)} DGGS cells")
        self.log_message(f"Prepared {len(self.country_grids)} countries for processing")

    def _create_merged_spatial_index(self):
        self.merged_spatial_index = self.merged_grid.sindex
        self.country_bounds = {gid: geom.bounds for gid, geom in self.country_geometries.items()}
        self.log_message(f"Created spatial index metadata for {len(self.country_bounds)} countries")

    def _prepare_country_grid_cache(self):
        cache_dir = os.path.join("temp", "country_grids_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.country_grid_pickle = {}
        for gid, gdf in self.country_grids.items():
            p = os.path.join(cache_dir, f"{gid}.pkl")
            gdf[['zoneID', 'geometry']].to_pickle(p)
            self.country_grid_pickle[gid] = p
        self.log_message(f"Prepared cached country grid files for {len(self.country_grid_pickle)} countries")

    def _get_sector_folders(self):
        sector_folders = []
        for item in os.listdir(self.edgar_folder):
            item_path = os.path.join(self.edgar_folder, item)
            if os.path.isdir(item_path) and item.endswith('_emi_nc'):
                sector_folders.append(item)
        sector_folders.sort()
        return sector_folders

    def _check_sector_year_file_exists(self, sector_folder, year):
        test_output_folder = os.path.join(os.getcwd(), "test", "test_EDGAR_csv", sector_folder)
        individual_filename = f"EDGAR_DGGS_methane_emissions_{sector_folder}_{year}.csv"
        return os.path.exists(os.path.join(test_output_folder, individual_filename))

    def _check_country_year_file_exists(self, sector_folder, gid, year):
        countries_folder = os.path.join(os.getcwd(), "test", "test_EDGAR_csv", sector_folder, "countries")
        country_filename = f"EDGAR_DGGS_methane_emissions_{sector_folder}_{gid}_{year}.csv"
        return os.path.exists(os.path.join(countries_folder, country_filename))

    def _save_country_df(self, df, sector_folder, gid, year):
        countries_folder = os.path.join(os.getcwd(), "test", "test_EDGAR_csv", sector_folder, "countries")
        os.makedirs(countries_folder, exist_ok=True)
        country_filename = f"EDGAR_DGGS_methane_emissions_{sector_folder}_{gid}_{year}.csv"
        country_path = os.path.join(countries_folder, country_filename)
        df.to_csv(country_path, index=False)
        return country_path

    def _get_files_by_year_for_sector(self, sector_folder):
        sector_path = os.path.join(self.edgar_folder, sector_folder)
        netcdf_files = [f for f in os.listdir(sector_path) if f.lower().endswith('.nc')]
        files_by_year = {}
        for filename in netcdf_files:
            # Use the LAST 4-digit token in filename as year (more robust)
            year_tokens = re.findall(r'(?<!\d)(\d{4})(?!\d)', filename)
            if not year_tokens:
                continue
            y = int(year_tokens[-1])
            if self.start_year <= y <= self.end_year:
                files_by_year[y] = os.path.join(sector_path, filename)
        return files_by_year

    def _generate_tasks_for_country(self, gid, grid_pickle_path, clipped, ipcc_code, year, sector_folder):
        data = clipped['data']
        mask = data > 0
        if not np.any(mask):
            return [], None
        rows, cols = np.where(mask)
        temp_dir = os.path.join("temp", "clipped_npy")
        os.makedirs(temp_dir, exist_ok=True)
        npy_path = os.path.join(temp_dir, f"clipped_{gid}_{ipcc_code}_{year}.npy")
        np.save(npy_path, data)
        n = len(rows)
        tasks = []
        chunk = self.chunk_size_pixels
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            tasks.append((gid, grid_pickle_path, npy_path, clipped['bounds'], clipped['transform'], rows[start:end], cols[start:end], ipcc_code, year, sector_folder))
        return tasks, npy_path

    def process_sector_year(self, sector_folder, year, netcdf_path):
        self.log_message(f"    Processing {sector_folder} - {year}")
        if self._check_sector_year_file_exists(sector_folder, year):
            self.log_message(f"      *** EXISTING SECTOR-YEAR FILE FOUND *** Skipping processing")
            try:
                test_output_folder = os.path.join(os.getcwd(), "test", "test_EDGAR_csv", sector_folder)
                individual_filename = f"EDGAR_DGGS_methane_emissions_{sector_folder}_{year}.csv"
                individual_path = os.path.join(test_output_folder, individual_filename)
                existing_results = pd.read_csv(individual_path)
                # Ensure Year column is present and filled
                if 'Year' not in existing_results.columns:
                    existing_results['Year'] = year
                else:
                    existing_results['Year'] = existing_results['Year'].fillna(year)
                self.log_message(f"      Loaded existing file: {individual_path} with {len(existing_results)} records")
                return existing_results
            except Exception as e:
                self.log_message(f"      Error loading existing sector-year file: {e}; will reprocess")

        sector = sector_folder.replace('_emi_nc', '')
        if sector not in self.sector_to_ipcc:
            self.log_message(f"      Warning: Sector {sector} not found in lookup table")
            return None
        ipcc_code = self.sector_to_ipcc[sector]
        self.log_message(f"      Mapped to IPCC2006 code: {ipcc_code}")

        try:
            ds = xr.open_dataset(netcdf_path)
            temp_tiff_path = _create_temp_raster_from_netcdf(ds)
        except Exception as e:
            self.log_message(f"      Error loading NetCDF {netcdf_path}: {e}")
            return None

        aggregate_maps = {}
        aggregate_targets = {}
        country_dataframes = []
        tasks = []
        temp_npy_paths = []

        to_process_gids = []
        for gid in self.country_grids.keys():
            if self._check_country_year_file_exists(sector_folder, gid, year):
                try:
                    countries_folder = os.path.join(os.getcwd(), "test", "test_EDGAR_csv", sector_folder, "countries")
                    country_filename = f"EDGAR_DGGS_methane_emissions_{sector_folder}_{gid}_{year}.csv"
                    country_path = os.path.join(countries_folder, country_filename)
                    df = pd.read_csv(country_path)
                    country_dataframes.append(df)
                    self.log_message(f"      Loaded existing country file for {gid}: {len(df)} records")
                except Exception as e:
                    self.log_message(f"      Error loading existing country file for {gid}: {e}; will reprocess")
                    to_process_gids.append(gid)
            else:
                to_process_gids.append(gid)

        for gid in to_process_gids:
            bounds = self.country_geometries[gid].bounds
            clipped = _clip_raster_by_bbox(temp_tiff_path, bounds)
            t, npy_path = self._generate_tasks_for_country(gid, self.country_grid_pickle[gid], clipped, ipcc_code, year, sector_folder)
            if t:
                tasks.extend(t)
                temp_npy_paths.append(npy_path)
                aggregate_maps[gid] = defaultdict(float)
                aggregate_targets[gid] = 0.0
            else:
                df = self.country_grids[gid][['zoneID']].copy().rename(columns={'zoneID': 'dggsID'})
                df['GID'] = gid
                df[ipcc_code] = 0.0
                df['Year'] = year
                df = df[df[ipcc_code] > 0]
                if len(df) > 0:
                    path = self._save_country_df(df, sector_folder, gid, year)
                    self.log_message(f"      Saved empty country file (no non-zero pixels) for {gid}: {path}")

        if tasks:
            num_proc = min(len(self.country_grids), multiprocessing.cpu_count(), self.max_processes)
            self.log_message(f"      Using {num_proc} parallel processes for {len(tasks)} pixel-chunk tasks across {len(to_process_gids)} countries")
            start = time.time()
            with multiprocessing.Pool(processes=num_proc) as pool:
                for result in pool.imap_unordered(_pixel_chunk_worker, tasks, chunksize=1):
                    gid, idx_array, val_array, chunk_target_sum, ipcc_code_ret, year_ret, sector_ret = result
                    if gid in aggregate_maps:
                        for idx, val in zip(idx_array, val_array):
                            aggregate_maps[gid][int(idx)] += float(val)
                        aggregate_targets[gid] += float(chunk_target_sum)
            self.log_message(f"      Completed chunk processing in {time.time() - start:.2f}s")

        try:
            if os.path.exists(temp_tiff_path):
                os.remove(temp_tiff_path)
            for p in temp_npy_paths:
                if p and os.path.exists(p):
                    os.remove(p)
        except Exception:
            pass

        for gid in to_process_gids:
            if gid not in aggregate_maps:
                continue
            contrib_map = aggregate_maps[gid]
            if not contrib_map:
                continue
            gdf = self.country_grids[gid]
            results = np.zeros(len(gdf))
            for idx, val in contrib_map.items():
                if 0 <= idx < len(results):
                    results[idx] = results[idx] + val
            total = float(np.sum(results))
            target_sum = float(aggregate_targets[gid])
            if total > 0.0 and target_sum > 0.0:
                scale = target_sum / total
                results = results * scale
                self.log_message(f"      {gid}: Applied scaling factor: {scale:.6f}")
            df = gdf[['zoneID']].copy().rename(columns={'zoneID': 'dggsID'})
            df['GID'] = gid
            df[ipcc_code] = results
            df['Year'] = year
            df = df[df[ipcc_code] > 0]
            if len(df) > 0:
                path = self._save_country_df(df, sector_folder, gid, year)
                self.log_message(f"      Saved country CSV for {gid}: {path} with {len(df)} records")
                country_dataframes.append(df)

        if not country_dataframes:
            self.log_message("      No country results produced")
            try:
                ds.close()
            except Exception:
                pass
            return None

        combined = pd.concat(country_dataframes, ignore_index=True)
        # Ensure Year is present and correct
        if 'Year' not in combined.columns:
            combined['Year'] = year
        else:
            combined['Year'] = combined['Year'].fillna(year)
        self.log_message(f"      Combined {len(combined)} records for sector {sector_folder} in {year}")
        test_output_folder = os.path.join(os.getcwd(), "test", "test_EDGAR_csv", sector_folder)
        os.makedirs(test_output_folder, exist_ok=True)
        individual_filename = f"EDGAR_DGGS_methane_emissions_{sector_folder}_{year}.csv"
        individual_path = os.path.join(test_output_folder, individual_filename)
        try:
            combined.to_csv(individual_path, index=False)
            self.log_message(f"      *** SAVED SECTOR-YEAR CSV *** {individual_path}")
            self.log_message(f"      File size: {os.path.getsize(individual_path)} bytes")
            self.log_message(f"      Records saved: {len(combined)}")
        except Exception as e:
            self.log_message(f"      Error saving sector-year CSV: {e}")
        try:
            ds.close()
        except Exception:
            pass
        return combined

    def process_sector(self, sector_folder):
        self.log_message(f"  Processing sector: {sector_folder}")
        files_by_year = self._get_files_by_year_for_sector(sector_folder)
        if not files_by_year:
            self.log_message(f"    No NetCDF files in year range for sector {sector_folder}")
            return None
        all_year_results = []
        for year in sorted(files_by_year.keys()):
            path = files_by_year[year]
            try:
                res = self.process_sector_year(sector_folder, year, path)
                if res is not None and len(res) > 0:
                    # Guarantee Year completeness before aggregating
                    if 'Year' not in res.columns:
                        res['Year'] = year
                    else:
                        res['Year'] = res['Year'].fillna(year)
                    all_year_results.append(res)
                    self.log_message(f"    Year {year}: {len(res)} records")
                else:
                    self.log_message(f"    Year {year}: No results for {os.path.basename(path)}")
            except Exception as e:
                self.log_message(f"    Error processing year {year} for {sector_folder}: {e}")
        if not all_year_results:
            self.log_message(f"  No results for sector {sector_folder}")
            return None
        combined = pd.concat(all_year_results, ignore_index=True)
        sector = sector_folder.replace('_emi_nc', '')
        ipcc_code = self.sector_to_ipcc.get(sector)
        if ipcc_code:
            combined = combined[combined[ipcc_code] > 0]
        return combined

    def process_all_sectors(self):
        start_time = time.time()
        self.log_message(f"Processing all {len(self.sector_folders)} sectors...")
        all_sector_results = []
        successful_sectors = []
        for sector_folder in self.sector_folders:
            self.log_message(f"  Processing sector: {sector_folder}")
            try:
                result = self.process_sector(sector_folder)
                if result is not None and len(result) > 0:
                    all_sector_results.append(result)
                    successful_sectors.append(sector_folder)
                    self.log_message(f"  Sector {sector_folder}: {len(result)} records")
                else:
                    self.log_message(f"  Sector {sector_folder}: No results")
            except Exception as e:
                self.log_message(f"  Error processing sector {sector_folder}: {e}")
        if not all_sector_results:
            self.log_message("No results found for any sector")
            return None
        combined_df = pd.concat(all_sector_results, ignore_index=True)
        # Ensure Year column exists and within requested range
        if 'Year' not in combined_df.columns:
            self.log_message("Warning: Combined dataframe missing Year column; filling with start_year")
            combined_df['Year'] = self.start_year
        combined_df['Year'] = combined_df['Year'].astype(int)
        combined_df = combined_df[(combined_df['Year'] >= self.start_year) & (combined_df['Year'] <= self.end_year)]

        # Combine to wide-by-IPCC, long-by-Year with no duplicates
        id_cols = ['dggsID', 'GID', 'Year']
        value_columns = [c for c in combined_df.columns if c not in id_cols]
        long_df = combined_df.melt(id_vars=id_cols, value_vars=value_columns, var_name='IPCC', value_name='value')
        long_df['value'] = long_df['value'].fillna(0.0)
        long_df = long_df.groupby(id_cols + ['IPCC'], as_index=False)['value'].sum()
        long_df = long_df[long_df['value'] > 0]
        wide_df = long_df.pivot_table(index=id_cols, columns='IPCC', values='value', aggfunc='sum', fill_value=0.0)
        wide_df = wide_df.reset_index()
        ipcc_cols = sorted([c for c in wide_df.columns if c not in id_cols])
        wide_df = wide_df[id_cols + ipcc_cols]

        output_filename = f"EDGAR_DGGS_methane_emissions_ALL_SECTORS_{self.start_year}_{self.end_year}.csv"
        output_path = os.path.join(self.output_folder, output_filename)
        wide_df.to_csv(output_path, index=False)
        self.log_message(f"\nFinal results saved to: {output_path}")
        self.log_message(f"Final output shape: {wide_df.shape}")
        total_time = time.time() - start_time
        self.log_message(f"\n{'='*60}")
        self.log_message("PROCESSING SUMMARY")
        self.log_message(f"{'='*60}")
        self.log_message(f"Year range processed: {self.start_year} to {self.end_year}")
        self.log_message(f"Total sectors processed: {len(successful_sectors)}/{len(self.sector_folders)}")
        self.log_message(f"Successful sectors: {', '.join(successful_sectors)}")
        self.log_message(f"Total output records: {len(combined_df)}")
        self.log_message(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        self.log_message(f"\nResuming capabilities:")
        self.log_message(f"  - Intermediate files saved to: test/test_EDGAR_csv/[sector]/countries/")
        self.log_message(f"  - Script will automatically skip existing files on restart")
        # Cleanup temp
        self._cleanup_temp_files()
        return output_path

    def _cleanup_temp_files(self):
        self.log_message("Cleaning up temporary files...")
        temp_folder = "temp"
        if os.path.exists(temp_folder):
            temp_files = [f for f in os.listdir(temp_folder) if f.startswith("temp_raster_") and f.endswith(".tiff")]
            for temp_file in temp_files:
                try:
                    os.remove(os.path.join(temp_folder, temp_file))
                except Exception:
                    pass
        clipped_npy_folder = os.path.join("temp", "clipped_npy")
        if os.path.exists(clipped_npy_folder):
            try:
                import shutil
                shutil.rmtree(clipped_npy_folder)
            except Exception:
                pass
        country_cache_folder = os.path.join("temp", "country_grids_cache")
        if os.path.exists(country_cache_folder):
            try:
                import shutil
                shutil.rmtree(country_cache_folder)
            except Exception:
                pass
        if os.path.exists(temp_folder):
            try:
                remaining = os.listdir(temp_folder)
                if not remaining:
                    os.rmdir(temp_folder)
            except Exception:
                pass
        self.log_message("Temporary file cleanup completed")


def main():
    edgar_folder = "E:/UCalgary_postdoc/data_source/GridInventory/1970-2022_EDGAR_v8.0_Greenhouse_Gas_CH4_Emissions"
    geojson_folder = "data/geojson"
    output_folder = "output"
    start_year = 2022
    end_year = 2022

    if not os.path.exists(edgar_folder):
        print(f"Error: EDGAR folder not found: {edgar_folder}")
        return
    if not os.path.exists(geojson_folder):
        print(f"Error: GeoJSON folder not found: {geojson_folder}")
        return

    converter = GlobalEDGARNetCDFToDGGSConverterOptimized(
        edgar_folder,
        geojson_folder,
        output_folder,
        start_year=start_year,
        end_year=end_year,
    )
    try:
        output_path = converter.process_all_sectors()
        converter.log_message(f"\nAll sectors conversion completed successfully!")
        converter.log_message(f"Combined output file: {output_path}")
    except Exception as e:
        converter.log_message(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        try:
            converter._cleanup_temp_files()
        except Exception:
            pass


if __name__ == "__main__":
    main()


