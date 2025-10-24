"""
Convert a single region GeoJSON to DGGS cells efficiently by tiling and streaming.

This refactor avoids loading the entire DGGS grid or full output into memory.
It splits the region's bbox into tiles, calls the DGGS CLI per-tile, filters
by intersection with the region geometry, and writes out incrementally.

Recommended output formats for very large results:
- Parquet dataset (fast, compact) — default when format is "parquet"
- Newline-delimited GeoJSON (geojsonl or geojsonl.gz) — streamable
"""

import os
import sys
import json
import math
import gzip
import argparse
import subprocess
from typing import Generator, Iterable, List, Tuple, Optional, Dict, Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape as shapely_shape
from shapely.prepared import prep as shapely_prep
from shapely import wkt as shapely_wkt
import concurrent.futures
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.errors import GEOSException
try:
    # Shapely 2.x
    from shapely.validation import make_valid as shapely_make_valid
except Exception:
    shapely_make_valid = None


GRID_TYPE_DEFAULT = "rhealpix"
LEVEL_DEFAULT = 7
INPUT_FILE_DEFAULT = os.path.join("data", "geojson", "europe.geojson")
OUTPUT_DIR_DEFAULT = os.path.join("data", "geojson", "regional_grid")


def run_dggs_grid(grid: str, level: int, bbox: str, timeout_sec: int = 600, retries: int = 2) -> Dict[str, Any]:
    """Run the DGGS CLI to generate a grid for the given bbox and return GeoJSON as dict.

    Note: We still parse tile output as a whole JSON document. Keep tiles small
    to limit memory use. If an NDJSON mode becomes available in the CLI, switch
    to streaming parse for even lower memory usage.
    """
    cmd = ["dgg", grid, "grid", str(level), "-bbox", bbox]
    attempt = 0
    last_err = None
    while attempt <= max(0, retries):
        try:
            print("Running DGGS CLI:", " ".join(cmd), f"(attempt {attempt+1}/{max(1, retries+1)})", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            if result.returncode != 0:
                last_err = RuntimeError(f"DGGS command failed (exit {result.returncode}): {result.stderr}")
                attempt += 1
                continue
            output = result.stdout.strip()
            if not output:
                last_err = RuntimeError("DGGS command returned empty output")
                attempt += 1
                continue
            return json.loads(output)
        except subprocess.TimeoutExpired as te:
            last_err = te
            print(f"DGGS CLI timed out after {timeout_sec}s for bbox {bbox}; retrying...")
            attempt += 1
        except Exception as e:
            last_err = e
            attempt += 1
    raise last_err if last_err is not None else RuntimeError("DGGS command failed for unknown reasons")


def generate_tiles(minx: float, miny: float, maxx: float, maxy: float, tile_deg: float) -> List[Tuple[float, float, float, float]]:
    """Generate non-overlapping lon/lat tiles covering the bbox.

    Returns tiles as (minx, miny, maxx, maxy) in lon/lat.
    """
    if tile_deg <= 0:
        return [(minx, miny, maxx, maxy)]

    tiles: List[Tuple[float, float, float, float]] = []
    # Clamp to bounds
    minx_c, miny_c = max(-180.0, minx), max(-90.0, miny)
    maxx_c, maxy_c = min(180.0, maxx), min(90.0, maxy)
    # Steps
    x_steps = max(1, math.ceil((maxx_c - minx_c) / tile_deg))
    y_steps = max(1, math.ceil((maxy_c - miny_c) / tile_deg))
    for yi in range(y_steps):
        ty_min = miny_c + yi * tile_deg
        ty_max = min(ty_min + tile_deg, maxy_c)
        for xi in range(x_steps):
            tx_min = minx_c + xi * tile_deg
            tx_max = min(tx_min + tile_deg, maxx_c)
            tiles.append((tx_min, ty_min, tx_max, ty_max))
    return tiles


def process_tile_to_rows(grid: str, level: int, tx_min: float, ty_min: float, tx_max: float, ty_max: float, region_wkt: str, region_name: str, timeout_sec: int, retries: int) -> List[Dict[str, Any]]:
    """Process a single tile: run DGGS, filter by intersection, and return rows for Parquet.

    The region geometry is passed as WKT to allow safe cross-process serialization.
    """
    bbox_str = f"{ty_min},{tx_min},{ty_max},{tx_max}"
    try:
        grid_geojson = run_dggs_grid(grid, level, bbox_str, timeout_sec=timeout_sec, retries=retries)
    except Exception as e:
        print(f"Warning: skipping tile bbox {bbox_str} due to DGGS error: {e}")
        return []
    region_geom = shapely_wkt.loads(region_wkt)
    prepared_region = shapely_prep(region_geom)
    rows: List[Dict[str, Any]] = []
    features = grid_geojson.get("features", [])
    for idx, feat in enumerate(features):
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shapely_shape(geom)
        if not prepared_region.intersects(shp):
            continue
        props = feat.get("properties", {}) or {}
        zone_id = props.get("zoneID")
        if zone_id is None:
            zone_id = props.get("id") or f"cell_{idx}"
        rows.append({
            "zoneID": zone_id,
            "region": region_name,
            "geometry": shp,
        })
    return rows


def process_tile_to_parquet(grid: str, level: int, tx_min: float, ty_min: float, tx_max: float, ty_max: float, region_wkt: str, region_name: str, dataset_dir: str, tile_idx: int, timeout_sec: int, retries: int) -> Tuple[str, int]:
    """Worker function: run DGGS, filter by intersection, write a parquet part, return (path, rows)."""
    os.makedirs(dataset_dir, exist_ok=True)
    bbox_str = f"{ty_min},{tx_min},{ty_max},{tx_max}"
    try:
        grid_geojson = run_dggs_grid(grid, level, bbox_str, timeout_sec=timeout_sec, retries=retries)
    except Exception as e:
        print(f"Warning: skipping tile #{tile_idx} bbox {bbox_str} due to DGGS error: {e}")
        part_path = os.path.join(dataset_dir, f"part-{tile_idx:05d}.parquet")
        gdf_empty = gpd.GeoDataFrame({"zoneID": [], "region": [], "geometry": []}, geometry="geometry", crs="EPSG:4326")
        gdf_empty.to_parquet(part_path, index=False)
        return part_path, 0
    region_geom = shapely_wkt.loads(region_wkt)
    prepared_region = shapely_prep(region_geom)
    rows: List[Dict[str, Any]] = []
    features = grid_geojson.get("features", [])
    for idx, feat in enumerate(features):
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shapely_shape(geom)
        if not prepared_region.intersects(shp):
            continue
        props = feat.get("properties", {}) or {}
        zone_id = props.get("zoneID")
        if zone_id is None:
            zone_id = props.get("id") or f"cell_{idx}"
        rows.append({
            "zoneID": zone_id,
            "region": region_name,
            "geometry": shp,
        })
    if not rows:
        part_path = os.path.join(dataset_dir, f"part-{tile_idx:05d}.parquet")
        # Write empty table with explicit string dtypes to avoid schema drift
        gdf_empty = gpd.GeoDataFrame(
            {
                "zoneID": pd.Series([], dtype="string"),
                "region": pd.Series([], dtype="string"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        gdf_empty.to_parquet(part_path, index=False)
        return part_path, 0
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    # Enforce stable types
    if "zoneID" in gdf.columns:
        gdf["zoneID"] = gdf["zoneID"].astype("string")
        gdf = gdf.drop_duplicates(subset=["zoneID"])  # within-tile dedup
    if "region" in gdf.columns:
        gdf["region"] = gdf["region"].astype("string")
    part_path = os.path.join(dataset_dir, f"part-{tile_idx:05d}.parquet")
    gdf.to_parquet(part_path, index=False)
    return part_path, len(gdf)


class GeoJSONLWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.gz = output_path.endswith(".gz")
        self.fh = gzip.open(output_path, "wt", encoding="utf-8") if self.gz else open(output_path, "w", encoding="utf-8")

    def write_features(self, features: Iterable[Dict[str, Any]]) -> None:
        for feat in features:
            self.fh.write(json.dumps(feat, separators=(",", ":")))
            self.fh.write("\n")

    def close(self) -> None:
        self.fh.close()


class GeoJSONStreamWriter:
    """Minimal streaming FeatureCollection writer to valid .geojson."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.gz = output_path.endswith(".gz")
        self.fh = gzip.open(output_path, "wt", encoding="utf-8") if self.gz else open(output_path, "w", encoding="utf-8")
        self.started = False
        self.count = 0
        # Write header
        self.fh.write('{"type":"FeatureCollection","features":[')

    def write_features(self, features: Iterable[Dict[str, Any]]) -> None:
        for feat in features:
            if self.started:
                self.fh.write(",")
            self.fh.write(json.dumps(feat, separators=(",", ":")))
            self.started = True
            self.count += 1

    def close(self) -> None:
        self.fh.write("]}")
        self.fh.close()


class ParquetPartitionWriter:
    def __init__(self, output_path: str, region: str, level: int):
        self.output_path = output_path
        self.region = region
        self.level = level
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.part_idx = 0
        self.all_rows = []
        # These will be configured by caller
        self.dataset_mode: bool = False
        self.dedup_enabled: bool = False
        self.seen_zone_ids: Optional[set] = None
        self.dataset_dir: Optional[str] = None

    def write_batch(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        if self.dataset_mode:
            if self.dataset_dir is None:
                base, _ = os.path.splitext(self.output_path)
                self.dataset_dir = base + "_dataset"
                os.makedirs(self.dataset_dir, exist_ok=True)
            # Deduplicate across parts if enabled
            if self.dedup_enabled and self.seen_zone_ids is not None:
                filtered: List[Dict[str, Any]] = []
                for r in rows:
                    zid = r.get("zoneID")
                    if zid is None:
                        filtered.append(r)
                    elif zid not in self.seen_zone_ids:
                        self.seen_zone_ids.add(zid)
                        filtered.append(r)
                rows = filtered
                if not rows:
                    return
            gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
            if "zoneID" in gdf.columns and not self.dedup_enabled:
                gdf = gdf.drop_duplicates(subset=["zoneID"])  # best-effort within-batch
            part_path = os.path.join(self.dataset_dir, f"part-{self.part_idx:05d}.parquet")
            gdf.to_parquet(part_path, index=False)
            self.part_idx += 1
        else:
            self.all_rows.extend(rows)

    def close(self) -> None:
        if self.dataset_mode:
            # Nothing to do; parts are already written
            return
        if self.all_rows:
            gdf = gpd.GeoDataFrame(self.all_rows, geometry="geometry", crs="EPSG:4326")
            if "zoneID" in gdf.columns:
                gdf = gdf.drop_duplicates(subset=["zoneID"])  # ensure unique cells
            gdf.to_parquet(self.output_path, index=False)


def iter_intersecting_features(grid_geojson: Dict[str, Any], prepared_region, region_name: str, seen_zone_ids: Optional[set]) -> Generator[Dict[str, Any], None, None]:
    """Yield DGGS features that intersect the region, adding metadata and de-duplicating by zoneID if provided."""
    features = grid_geojson.get("features", [])
    for idx, feat in enumerate(features):
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shapely_shape(geom)
        if not prepared_region.intersects(shp):
            continue
        props = feat.get("properties", {}) or {}
        zone_id = props.get("zoneID")
        if zone_id is None:
            # Fallback if upstream does not provide zoneID
            zone_id = props.get("id") or f"cell_{idx}"
            props["zoneID"] = zone_id
        if seen_zone_ids is not None:
            if zone_id in seen_zone_ids:
                continue
            seen_zone_ids.add(zone_id)
        props["region"] = region_name
        feat["properties"] = props
        yield feat


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert region GeoJSON to DGGS cells by tiling and streaming.")
    parser.add_argument("--grid", default=GRID_TYPE_DEFAULT, help="DGGS grid type (default: rhealpix)")
    parser.add_argument("--level", type=int, default=LEVEL_DEFAULT, help="DGGS resolution level")
    parser.add_argument("--input", dest="input_path", default=INPUT_FILE_DEFAULT, help="Input region GeoJSON path")
    parser.add_argument("--output-dir", dest="output_dir", default=OUTPUT_DIR_DEFAULT, help="Output directory")
    parser.add_argument("--format", dest="out_format", default="parquet", choices=["parquet", "geojsonl", "geojsonl.gz", "geojson", "geojson.gz"], help="Output format")
    parser.add_argument("--tile-deg", type=float, default=2.0, help="Tile size in degrees (lon/lat)")
    parser.add_argument("--batch-size", type=int, default=10000, help="Number of features per write batch")
    parser.add_argument("--dedup", action="store_true", help="De-duplicate cells across tiles by zoneID (recommended)")
    parser.add_argument("--max-tiles", type=int, default=None, help="Process at most N tiles (for testing)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers for tiles (Parquet only)")
    parser.add_argument("--parquet-dataset", action="store_true", help="Write Parquet as multiple part files to reduce memory usage")
    parser.add_argument("--timeout-sec", type=int, default=600, help="Timeout in seconds for each DGGS CLI call")
    parser.add_argument("--retries", type=int, default=2, help="Number of retries for failed/timeout DGGS calls")

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Input not found: {args.input_path}")
        return 1

    region_name = os.path.splitext(os.path.basename(args.input_path))[0]

    # Read region geometry and ensure WGS84
    region_gdf = gpd.read_file(args.input_path)
    if region_gdf.empty:
        print("Input GeoJSON is empty")
        return 1
    if region_gdf.crs is None:
        region_gdf = region_gdf.set_crs("EPSG:4326")
    else:
        region_gdf = region_gdf.to_crs("EPSG:4326")

    # Clean invalid geometries to avoid TopologyException during union
    def _clean_geom(geom):
        if geom is None or geom.is_empty:
            return None
        try:
            if not geom.is_valid:
                if shapely_make_valid is not None:
                    geom = shapely_make_valid(geom)
                else:
                    geom = geom.buffer(0)
        except Exception:
            try:
                geom = geom.buffer(0)
            except Exception:
                return None
        return geom if (geom is not None and not geom.is_empty) else None

    region_gdf = region_gdf[region_gdf.geometry.notnull()]
    region_gdf["geometry"] = region_gdf.geometry.apply(_clean_geom)
    region_gdf = region_gdf[region_gdf.geometry.notnull() & ~region_gdf.geometry.is_empty]

    # Union and prepare geometry for fast spatial predicates, with robust fallback
    try:
        region_geom = region_gdf.geometry.union_all()
    except (GEOSException, Exception):
        region_geom = gpd.GeoSeries(region_gdf.geometry.values, crs="EPSG:4326").unary_union
    # Ensure union result is valid
    try:
        if not region_geom.is_valid:
            region_geom = shapely_make_valid(region_geom) if shapely_make_valid is not None else region_geom.buffer(0)
    except Exception:
        pass
    prepared_region = shapely_prep(region_geom)

    # Determine tiles
    minx, miny, maxx, maxy = region_geom.bounds  # lon/lat
    tiles = generate_tiles(minx, miny, maxx, maxy, args.tile_deg)
    if args.max_tiles is not None:
        tiles = tiles[: args.max_tiles]
    num_tiles = len(tiles)
    print(f"Region bbox (lon/lat): {minx:.6f},{miny:.6f} to {maxx:.6f},{maxy:.6f}; tiles: {num_tiles} @ {args.tile_deg}°")

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    total_kept = 0
    seen_zone_ids: Optional[set] = set() if args.dedup else None

    # Writer setup
    writer_geojsonl: Optional[GeoJSONLWriter] = None
    writer_geojson: Optional[GeoJSONStreamWriter] = None
    writer_parquet: Optional[ParquetPartitionWriter] = None

    if args.out_format == "parquet":
        out_path = os.path.join(args.output_dir, f"{region_name}_grid_res{args.level}.parquet")
        writer_parquet = ParquetPartitionWriter(out_path, region_name, args.level)
        # Attach dataset/dedup configuration dynamically
        writer_parquet.dataset_mode = bool(args.parquet_dataset)
        writer_parquet.dedup_enabled = bool(args.dedup)
        writer_parquet.seen_zone_ids = set() if writer_parquet.dataset_mode and writer_parquet.dedup_enabled else None
        print(f"Writing Parquet file to: {out_path} ({'dataset mode' if writer_parquet.dataset_mode else 'single file'})")
    elif args.out_format in ("geojsonl", "geojsonl.gz"):
        ext = ".geojsonl.gz" if args.out_format.endswith(".gz") else ".geojsonl"
        out_path = os.path.join(args.output_dir, f"{region_name}_res{args.level}{ext}")
        writer_geojsonl = GeoJSONLWriter(out_path)
        print(f"Writing GeoJSONL to: {out_path}")
    else:
        # geojson or geojson.gz
        ext = ".geojson.gz" if args.out_format.endswith(".gz") else ".geojson"
        out_path = os.path.join(args.output_dir, f"{region_name}_res{args.level}{ext}")
        writer_geojson = GeoJSONStreamWriter(out_path)
        print(f"Writing GeoJSON FeatureCollection to: {out_path}")

    # Process tiles
    batch_rows: List[Dict[str, Any]] = []  # for parquet
    batch_feats: List[Dict[str, Any]] = []  # for geojson/geojsonl

    workers = args.workers if args.workers and args.workers > 0 else None
    if writer_parquet is not None and writer_parquet.dataset_mode and workers and workers > 1:
        print(f"Parallel tile processing enabled with {workers} workers (direct part writes)", flush=True)
        # Ensure dataset dir name <region>_grid_res<level>_dataset
        base, _ = os.path.splitext(writer_parquet.output_path)
        dataset_dir = base + "_dataset"
        os.makedirs(dataset_dir, exist_ok=True)
        region_wkt = region_geom.wkt
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures_list = []
            for tile_idx, (tx_min, ty_min, tx_max, ty_max) in enumerate(tiles):
                print(f"Tile {tile_idx+1}/{num_tiles}: bbox (lon/lat) {tx_min:.4f},{ty_min:.4f} to {tx_max:.4f},{ty_max:.4f}", flush=True)
                futures_list.append(executor.submit(
                    process_tile_to_parquet,
                    args.grid,
                    args.level,
                    tx_min,
                    ty_min,
                    tx_max,
                    ty_max,
                    region_wkt,
                    region_name,
                    dataset_dir,
                    tile_idx,
                    int(args.timeout_sec),
                    int(args.retries),
                ))
            for fut in concurrent.futures.as_completed(futures_list):
                part_path, n_rows = fut.result()
                total_kept += n_rows
                print(f"  wrote {os.path.basename(part_path)}; rows in part: {n_rows}; total kept: {total_kept}", flush=True)
        # Wire the writer to point to the dataset dir we just wrote
        writer_parquet.dataset_dir = dataset_dir
    else:
        for tile_idx, (tx_min, ty_min, tx_max, ty_max) in enumerate(tiles):
            # DGGS bbox expects south,west,north,east (lat_min, lon_min, lat_max, lon_max)
            bbox_str = f"{ty_min},{tx_min},{ty_max},{tx_max}"
            print(f"Tile {tile_idx+1}/{num_tiles}: bbox (lon/lat) {tx_min:.4f},{ty_min:.4f} to {tx_max:.4f},{ty_max:.4f}", flush=True)

            try:
                grid_geojson = run_dggs_grid(args.grid, args.level, bbox_str, timeout_sec=int(args.timeout_sec), retries=int(args.retries))
            except Exception as e:
                print(f"Warning: skipping tile bbox {bbox_str} due to DGGS error: {e}")
                continue

            # Iterate intersecting features and write by batches
            for feat in iter_intersecting_features(grid_geojson, prepared_region, region_name, seen_zone_ids):
                if writer_parquet is not None:
                    props = feat.get("properties", {}) or {}
                    zone_id = props.get("zoneID")
                    batch_rows.append({
                        "zoneID": zone_id,
                        "region": props.get("region", region_name),
                        "geometry": shapely_shape(feat["geometry"]),
                    })
                    if len(batch_rows) >= args.batch_size:
                        writer_parquet.write_batch(batch_rows)
                        total_kept += len(batch_rows)
                        print(f"  wrote parquet batch; total kept: {total_kept}")
                        batch_rows.clear()
                else:
                    batch_feats.append(feat)
                    if len(batch_feats) >= args.batch_size:
                        if writer_geojsonl is not None:
                            writer_geojsonl.write_features(batch_feats)
                        elif writer_geojson is not None:
                            writer_geojson.write_features(batch_feats)
                        total_kept += len(batch_feats)
                        print(f"  wrote json batch; total kept: {total_kept}")
                        batch_feats.clear()

            # Free per-tile JSON once processed
            grid_geojson = None  # hint for GC

    # Flush remaining
    if writer_parquet is not None and batch_rows:
        writer_parquet.write_batch(batch_rows)
        total_kept += len(batch_rows)
        batch_rows.clear()
    if (writer_geojsonl is not None or writer_geojson is not None) and batch_feats:
        if writer_geojsonl is not None:
            writer_geojsonl.write_features(batch_feats)
        elif writer_geojson is not None:
            writer_geojson.write_features(batch_feats)
        total_kept += len(batch_feats)
        batch_feats.clear()

    # Close writers
    if writer_parquet is not None:
        writer_parquet.close()
        # If dataset mode, combine parts into a single Parquet and clean up
        if getattr(writer_parquet, "dataset_mode", False) and writer_parquet.dataset_dir is not None:
            combined_path = writer_parquet.output_path
            part_files = sorted([
                os.path.join(writer_parquet.dataset_dir, f)
                for f in os.listdir(writer_parquet.dataset_dir)
                if f.endswith(".parquet")
            ])
            if not part_files:
                print("No parts to combine; dataset directory is empty")
            else:
                print(f"Combining {len(part_files)} part files into single Parquet: {combined_path}")
                # Initialize writer with schema and geo metadata from first part
                first_table = pq.read_table(part_files[0])
                schema = first_table.schema
                metadata = schema.metadata
                writer = pq.ParquetWriter(combined_path, schema=schema, version="2.6", compression="snappy")
                try:
                    seen_ids: Optional[set] = set()
                    for idx, p in enumerate(part_files, start=1):
                        tbl = pq.read_table(p)
                        # Coerce dtypes to match the first table schema to avoid schema drift
                        target_schema = schema
                        # Ensure zoneID/region are strings
                        col_names = tbl.schema.names
                        if "zoneID" in col_names and not pa.types.is_string(tbl.schema.field("zoneID").type):
                            tbl = tbl.set_column(col_names.index("zoneID"), "zoneID", tbl.column("zoneID").cast(pa.string()))
                        if "region" in col_names and not pa.types.is_string(tbl.schema.field("region").type):
                            tbl = tbl.set_column(col_names.index("region"), "region", tbl.column("region").cast(pa.string()))
                        # Align to target schema order/types where possible
                        try:
                            tbl = tbl.cast(target_schema)
                        except Exception:
                            pass
                        # Deduplicate by zoneID across parts
                        try:
                            zid_col = tbl.column("zoneID")
                        except KeyError:
                            zid_col = None
                        if zid_col is not None:
                            zid_list = zid_col.to_pylist()
                            keep_indices: List[int] = []
                            for i, zid in enumerate(zid_list):
                                if zid is None:
                                    keep_indices.append(i)
                                elif zid not in seen_ids:
                                    seen_ids.add(zid)
                                    keep_indices.append(i)
                            if len(keep_indices) < tbl.num_rows:
                                if len(keep_indices) == 0:
                                    # Make an empty table with same schema
                                    tbl = tbl.slice(0, 0)
                                else:
                                    idx_array = pa.array(keep_indices, type=pa.int64())
                                    tbl = tbl.take(idx_array)
                        if tbl.num_rows > 0:
                            writer.write_table(tbl)
                        print(f"  combined part {idx}/{len(part_files)}; rows written so far")
                finally:
                    writer.close()
                # Restore geo metadata if lost (parquet writer preserves schema metadata)
                if metadata is not None:
                    # Re-open and set file metadata is non-trivial; skip explicit reset
                    pass
                # Clean up dataset directory
                try:
                    shutil.rmtree(writer_parquet.dataset_dir)
                    print(f"Removed intermediate dataset directory: {writer_parquet.dataset_dir}")
                except Exception as e:
                    print(f"Warning: failed to remove dataset directory {writer_parquet.dataset_dir}: {e}")
    if writer_geojsonl is not None:
        writer_geojsonl.close()
    if writer_geojson is not None:
        writer_geojson.close()

    print(f"Done. Total DGGS cells kept: {total_kept}")
    return 0 if total_kept > 0 else 1


if __name__ == "__main__":
    sys.exit(main())


