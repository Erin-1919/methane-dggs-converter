# CLAUDE.md

## What this repo is

A data-harmonization pipeline that converts heterogeneous **gridded methane emission inventories** (13 open-access products, global → sub-national) into the **rHEALPix Discrete Global Grid System (DGGS)**, standardizing four things at once:

| Dimension | Harmonized to |
|---|---|
| Spatial framework | rHEALPix DGGS (equal-area cells, radix-9 refinement) |
| CRS | EPSG:4326 before rasterization |
| Sector classification | IPCC 2006 codes |
| Unit | Mg a⁻¹ (megagrams CH₄ per year) **per cell** |

Output is tabular: one row = (`dggsID`, `Year`), one column per IPCC 2006 category, plus `GID` (ISO 3166-1 alpha-3) for global inventories. Cell geometries are **not** embedded — they live separately as GeoParquet/GeoJSON and are joined on `dggsID`.

This repo is the software behind the manuscript *"Harmonized Global-to-Regional Gridded Methane Inventories in a Discrete Global Grid Framework"* (Li, Gao, Liang — IJGI 2026 submission). Published dataset: https://doi.org/10.5281/zenodo.17362125.

## Where the final data lives

**`E:\UCalgary_postdoc\genAI_dggs_ch4\methane_final_dataset\`** — the authoritative, published outputs (mirrored on Zenodo, DOI above). This is *not* the repo's `output/` folder, which holds working/intermediate runs with inconsistent names. When asked about "the data" or "the final dataset", use this folder and these filenames:

| File | Level | Years | `GID`? |
|---|---|---|---|
| `EDGAR_DGGS_methane_emissions_ALL_SECTORS_1970-2022.csv` (~9.4 GB; also split per year in `EDGAR/`) | 6 | 1970–2022 | yes |
| `GFEI_DGGS_methane_emissions_2016-2019-2020.csv` | 6 | 2016/2019/2020 | yes |
| `CAMS-REG_DGGS_methane_emissions_2005-2022.csv` | **7** | 2005–2022 | no |
| `NYS_DGGS_methane_emissions_2020.csv` | **10** | 2020 | no |
| `Switzerland_DGGS_methane_emissions_2011.csv` | 9 | 2011 | no |
| `US_DGGS_methane_emissions_2012-2018.csv` | 6 | 2012–2018 | no |
| `US_OG_DGGS_methane_emissions_2021.csv` | 6 | 2021 | no |
| `Canada_DGGS_methane_emissions_2018.csv` | 6 | 2018 | no |
| `Mexico_DGGS_methane_emissions_2015.csv` | 6 | 2015 | no |
| `China_DGGS_methane_emissions_1990-2020.csv` | 6 | 1990–2020 | no |
| `China_SACMS_DGGS_methane_emissions_2011.csv` | 6 | 2011 | no |
| `CMS_Canada_DGGS_methane_emissions_2013.csv` / `CMS_Mexico_..._2010.csv` | 6 | 2013 / 2010 | no |
| `India_Coal_..._2018.csv` / `Australia_Coal_..._2018.csv` | 6 | 2018 | no |

- **`sampleData/`** holds a `*_sample.csv` for each dataset — use these for schema inspection and any exploratory work. Never `head`/load the full EDGAR or CAMS-REG CSVs (9.4 GB / 880 MB) without chunking.
- Cell geometries are **not** in this folder — GeoParquet geometry files per resolution are in the Zenodo archive (and, for working runs, under the repo's gitignored `data/geojson/`).
- The paper's resolution levels are authoritative: CAMS-REG Europe = **7**, NYS = **10**. Some repo working files (`Europe_..._res6.csv`, `NYS_..._res7.csv`, entries in `analysis_results/configs/dggs_resolution_map.json`) are earlier trial runs at other levels — don't trust them over the table above.

**Name note:** "ARC" in the repo name = University of Calgary **Advanced Research Computing** cluster (SLURM), where all production runs happen. This is not a local-laptop pipeline.

## Non-obvious design decisions (read before changing anything)

1. **Mass preservation is the contract.** Every converter ends with an explicit scaling step: `scaling_factor = total_raster_value / total_weighted_value`, applied to the DGGS results. This is what keeps pre/post totals equal. Don't remove it as "redundant" — the area-weighted allocation alone does not close exactly.
2. **Scaling is applied per (country, IPCC code, year)** for global inventories. Consequence, documented in the paper: country totals are preserved exactly, but global re-aggregation accumulates boundary/rounding effects → relative differences of ~1% (EDGAR 1970–2022, mean 1.03%) and 2.6–3.1% (GFEI). Single-country datasets are ≪0.1%. These are expected values, not bugs.
3. **Raster-first, not cell-first.** Non-zero raster pixels are filtered first, then their overlapping DGGS cells are found via bounding-box pre-filter + spatial index. Reversing this (iterating cells) blows up runtime by orders of magnitude.
4. **Per-area inputs are integrated over the *source* pixel area before redistribution** — never after. Pixel areas come from the affine transform (projected data), spherical geometry, or the source file's own area variable (see `data/area_npy/`).
5. **DGGS grids are pre-computed, not generated inside converters.** Grid creation (Phase 1) shells out to the DGGAL `dgg` CLI (`dgg rhealpix grid <level> -bbox ...`) via `subprocess`; converters just load the resulting `.parquet`/`.geojson`. The `dggal` pip package is in `environment.yml`; the `dgg` binary must be on `PATH`.
6. **Resolution level is chosen per inventory to match native resolution** — not one global level:
   - level 10 (~0.024 km²) → New York State, 100 m
   - level 9 (~0.22 km²) → Switzerland, 500 m
   - level 7 (~17.8 km²) → CAMS-REG Europe, 0.05°×0.10°
   - level 6 (~160 km²) → everything at 0.1°×0.1°, **and** China SACMS at 0.25° (deliberately finer than level 5 to avoid over-smoothing)
   Canonical map: `analysis_results/configs/dggs_resolution_map.json`.
7. **Pre-aggregated "total" fields in source products are dropped** before redistribution to avoid double counting with per-sector fields.
8. **All-zero rows are dropped** from outputs (file size / query performance). Cell absence therefore means "no emissions recorded", not "outside domain" — use the coverage GeoJSONs in `scripts/analysis/create_dggs_coverage_geojsons.py` for extent.
9. **IPCC 2006 codes cannot be grouped by naive string-prefix matching** in general — the scheme mixes Roman numerals (`1B2aiii1`, `3A1ai`). The first-digit split into Energy/IPPU/AFOLU/Waste (used in `compute_sectoral_breakdown.py`) is safe; deeper prefix logic is not. Also, **some columns are `+`-joined multi-code names** where a source variable maps to several IPCC categories that couldn't be separated — e.g. EDGAR's `1A1b+1A1ci+1A1cii+1A5biii+1B1b+1B2aiii6+1B2biii3+1B1c` and CAMS-REG's `1A3c+1A4cii+1A3e+1A5b`. Any code-matching logic must split on `+` and handle a column belonging to multiple groups.
10. **DGGS cells receive emissions from any overlapping pixel** regardless of land/water classification. Offshore grids are generated separately and merged with country grids, with duplicate `zoneID` removal.

## Layout

```
scripts/
  dggs_grid_creation/   Phase 1 — build rHEALPix grids from country/offshore boundaries (pygadm → simplify → dgg CLI)
  netcdf_conversion/    Per-inventory NetCDF → DGGS converters (11 scripts; one class per inventory)
  geotiff_conversion/   China CHN-CH₄ GeoTIFF time series (1990–2020)
  csv_conversion/       India/Australia coal-mine point CSVs
  utilities/            Combine/merge/dedupe grid geometries
  analysis/             Post-hoc summaries: dataset summary table, sectoral breakdown, coverage geometries
  test_scripts/         GITIGNORED. compare_*_before_after_totals.py = the validation harness;
                        visualize_*.py = paper figures; test_*.py = exploratory; archive_* = superseded
SLURM_job_scripts/      One sbatch script per conversion; sets NUM_CORES + OMP_NUM_THREADS=1
data/lookup/            Committed. variable → IPCC2006 crosswalks, CRT↔IPCC, IPCC 1996→2006 mappings
data/geojson/           GITIGNORED. Pre-computed DGGS grids (.geojson + .parquet)
data/area_npy/          Source-provided pixel areas for CMS/GFEI/Mexico
output/, log/, temp/, test/   GITIGNORED
analysis_results/       Committed. Summary CSVs + configs/*.json
paper/                  GITIGNORED. Manuscript .docx + figures
```

## Converter anatomy

Every converter is a single class with the same skeleton. When adding an inventory, copy the closest existing one rather than inventing structure:

1. `_setup_logging()` → timestamped log in `log/`, file + console handlers
2. `_load_ipcc_lookup()` → `data/lookup/<name>_variable_lookup.csv`, requires columns `variable` + `IPCC2006`
3. load DGGS grid (`gpd.read_parquet`, must have `zoneID`) → `_create_spatial_index()` (bounds list)
4. `aggregate_variables_by_ipcc_code()` → sum source variables sharing an IPCC code
5. `convert_aggregated_to_raster()` → xarray/rasterio raster + affine transform + pixel areas
6. unit conversion → Mg a⁻¹ (formulas below)
7. `calculate_weighted_values_raster_first()` per IPCC code, parallelized over `NUM_CORES` (env var, default 8) with `multiprocessing`
8. scaling factor applied
9. rename `zoneID` → `dggsID`, drop all-zero rows, write per-year CSV, then combined `*_ALL_FILES.csv`
10. **resume capability**: existing per-year/per-country CSVs are detected and reloaded rather than recomputed — relied on heavily for multi-day EDGAR runs

### Unit conversions (must stay mass-conservative)
- `Mg km⁻² a⁻¹` → `V × 1e-6 × A_pixel_m2`
- `kg m⁻² s⁻¹` → `V × A_pixel_m2 × 31_536_000 × 1e-3`
- `kg h⁻¹` → `V × 8760 × 1e-3` (already a total; no area term)
- `g m⁻² a⁻¹` → `V × A_pixel_m2 × 1e-6`
- `molec CH₄ cm⁻² s⁻¹` → `(V × A_pixel_cm2 × 31_536_000 / 6.022e23) × 16.04 × 1e-6`
- `t a⁻¹` → unchanged (1 t = 1 Mg)

## Environment & running

- Conda env **`netcdf_dggs_converter`** (`environment.yml`, Python 3.9): geopandas, rasterio, xarray, netcdf4, gdal/proj/geos, numba, dask, + pip `dggal`. **Don't install packages — report the missing package and the command.**
- Scripts use **relative paths from the repo root** for repo data (`data/lookup/...`) and **hardcoded absolute HPC paths** for source inventories (`/home/mingke.li/GridInventory/...`) in `main()`. Editing those paths is the normal way to point a converter at new data; there is no CLI arg parsing.
- No test framework, no linter config, no CI. Validation = run the matching `scripts/test_scripts/compare_*_before_after_totals.py` and check the relative difference in `analysis_results/before_after_comprison/`.
- Local (Windows) work is realistically limited to analysis scripts and small regional datasets; global conversions need the cluster (EDGAR: 24 cores, 60 GB, 48 h).

```bash
# Phase 1, once
python scripts/dggs_grid_creation/create_global_country_geojson.py
python scripts/dggs_grid_creation/simplify_global_countries.py
python scripts/dggs_grid_creation/convert_country_geojson_to_dggs.py
python scripts/dggs_grid_creation/convert_offshore_to_dggs.py
python scripts/utilities/merge_country_offshore_dggs_geometries.py
python scripts/utilities/combine_geojson_folder.py

# Phase 2, per inventory (HPC)
sbatch SLURM_job_scripts/run_edgar_netcdf_conversion.sh
```

## Conventions to follow

- **Column names:** `dggsID` in outputs (`zoneID` only inside grid geometry files), `Year`, `GID`, IPCC codes verbatim as column headers.
- **Output naming:** `<Source>_DGGS_methane_emissions_<year|ALL_FILES>[_res<N>].csv`.
- New converters: add the script, a `data/lookup/<name>_variable_lookup.csv`, a `SLURM_job_scripts/run_<name>_conversion.sh`, a `compare_<name>_before_after_totals.py`, and entries in `analysis_results/configs/{dggs_resolution_map,year_range_map}.json`.
- Logging: use the `log_message()` pattern (print + logger) — the logs are the only progress signal on long SLURM jobs.
- Don't commit anything under `output/`, `data/geojson/`, `log/`, `temp/`, `test/`, `scripts/test_scripts/`, or `paper/` (all gitignored). `data/lookup/*.csv` and `analysis_results/` **are** tracked.

## Known rough edges

- The repo's `output/` is a working directory, not a deliverable: `output/New folder/`, `Europe_..._res6.csv`, `NYS_..._res7.csv` and `*_ALL_FILES.csv` names are all superseded by `methane_final_dataset/` (see above). `analysis_results/before_after_comprison/` has a typo in its name (kept — scripts write to it).
- Converters share ~80% of their logic by copy-paste, not by a common module. Refactoring is tempting but risky: per-inventory quirks (area sources, time-index→year mapping, CRS, unit) are embedded in those copies. If asked to refactor, extract cautiously and re-run the before/after comparisons as the regression test.
- The paper credits a Python package **`uraster`** for the raster-first area-weighted redistribution; this repo implements that logic inline in the converters.
