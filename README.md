# Methane Grid Calculation with DGGS

This project converts various NetCDF methane emission datasets to Discrete Global Grid System (DGGS) format using pre-calculated country-specific DGGS grids.

## Project Structure

The project is organized into three main script categories:

### 📁 `scripts/dggs_grid_creation/`
Scripts for creating and preparing DGGS grids for all countries:

1. **`create_global_country_geojson.py`**
   - Creates global country boundaries from pygadm
   - Outputs: `data/geojson/global_countries.geojson`

2. **`simplify_global_countries.py`**
   - Simplifies country geometries to reduce file size
   - Outputs: `data/geojson/global_countries_simplify.geojson`

3. **`convert_country_geojson_to_dggs.py`**
   - Converts country boundaries to DGGS grid cells
   - Uses rhealpix grid type at resolution 6
   - Outputs: Individual country DGGS files

4. **`simplify_country_dggs_geometries.py`**
   - Simplifies DGGS grid geometries for countries
   - Reduces file size while maintaining spatial accuracy

5. **`simplify_offshore_dggs_geometries.py`**
   - Simplifies DGGS grid geometries for offshore areas
   - Handles offshore-specific geometry processing

### 📁 `scripts/utilities/`
Utility scripts for combining and merging data:

6. **`combine_geojson_folder.py`**
   - Combines individual country DGGS files into a single file
   - Outputs: `data/geojson/global_countries_dggs_merge.geojson`

7. **`merge_country_offshore_dggs_geometries.py`**
   - Merges country and offshore DGGS grids
   - Handles duplicate zoneID removal

### 📁 `scripts/netcdf_conversion/`
Scripts for converting NetCDF data to DGGS using pre-calculated grids:

8. **`canada_netcdf_to_dggs_converter.py`**
   - Converts Canada NetCDF methane data to DGGS
   - Uses IPCC2006 code aggregation
   - Handles 2018 Canada anthropogenic methane emissions

9. **`global_edgar_netcdf_to_dggs_optimize.py`**
   - Converts EDGAR global NetCDF data to DGGS
   - Optimized for large-scale processing
   - Handles 1970-2022 EDGAR v8.0 greenhouse gas CH4 emissions

10. **`global_gfei_netcdf_to_dggs_optimize.py`**
    - Converts GFEI global NetCDF data to DGGS
    - Handles 2016-2020 Global Fuel Exploitation Inventory
    - Multi-year processing capability

11. **`mexico_netcdf_to_dggs_converter.py`**
    - Converts Mexico NetCDF methane data to DGGS
    - Uses area data from Canada files
    - Handles 2015 Mexico anthropogenic methane emissions

12. **`us_netcdf_to_dggs_converter.py`**
    - Converts US NetCDF methane data to DGGS
    - Handles flux units (molecules CH₄ cm⁻² s⁻¹)
    - Processes 2012-2018 US anthropogenic methane emissions

## Workflow

### Phase 1: DGGS Grid Creation (Pre-calculated Grids)
1. Create global country boundaries
2. Simplify country geometries
3. Convert to DGGS grid cells
4. Simplify DGGS geometries
5. Combine and merge grids

### Phase 2: NetCDF to DGGS Conversion
1. Use pre-calculated DGGS grids
2. Convert NetCDF data to raster format
3. Apply area-weighted distribution to DGGS cells
4. Output CSV files with DGGS cell values

## Data Structure

### Input Data
- **NetCDF files**: Various methane emission datasets
- **Lookup tables**: IPCC2006 code mappings in `data/lookup/`
- **Area data**: Grid cell area information in `data/area_npy/`

### Output Data
- **DGGS grids**: Country-specific DGGS cell geometries
- **CSV files**: DGGS cell values with emission data
- **Logs**: Processing logs in `log/` directory

## Key Features

- **Pre-calculated grids**: Efficient processing using pre-computed DGGS grids
- **Area-weighted distribution**: Accurate spatial allocation of emission values
- **IPCC2006 aggregation**: Standardized emission categorization
- **Multi-source support**: Handles various data formats and units
- **Parallel processing**: Optimized for large-scale data processing
- **Resume capability**: Can restart from intermediate results

## Usage

1. **Create DGGS grids** (run once):
   ```bash
   python scripts/dggs_grid_creation/create_global_country_geojson.py
   python scripts/dggs_grid_creation/simplify_global_countries.py
   python scripts/dggs_grid_creation/convert_country_geojson_to_dggs.py
   # ... continue with other grid creation scripts
   ```

2. **Convert NetCDF data** (run as needed):
   ```bash
   python scripts/netcdf_conversion/canada_netcdf_to_dggs_converter.py
   python scripts/netcdf_conversion/global_edgar_netcdf_to_dggs_optimize.py
   # ... run other conversion scripts
   ```

