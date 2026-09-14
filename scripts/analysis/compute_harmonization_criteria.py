# -*- coding: utf-8 -*-
"""Measure the harmonization criteria reported in the paper (revision item C13).

Three of the five criteria are computed here. Mass conservation is already reported in
Tables 4 and 5, and semantic coverage comes from the sector crosswalk.

  area comparability   coefficient of variation of native source-pixel area across each
                       inventory's domain, against zero for an equal-area DGGS. For a
                       graticule source the area follows the latitude. For a projected
                       source it follows the projection: an equal-area conic holds it
                       constant, while a conformal projection lets it drift slightly, so
                       the projected sources are measured rather than assumed.
  aggregation exactness residual of a prefix roll-up. Truncating one character from the
                       cell identifier names the parent cell, so aggregating to a coarser
                       level is a summation and introduces no geometric error. The
                       residual reported is the relative difference between the rolled-up
                       total and the original total.
  resolution fidelity  DGGS cell area divided by the mean native pixel area, which states
                       how closely the chosen level matches the source resolution.

Run from the repo root:
    conda run -n geom python scripts/analysis/compute_harmonization_criteria.py
"""
import numpy as np
import pandas as pd

DATA = 'E:/UCalgary_postdoc/genAI_dggs_ch4/methane_final_dataset/'
CENTROIDS = 'data/geojson/global_countries_dggs_merge_centroids.geojson'
EUROPE_GEOM = 'data/geojson/europe_dggs_geom_res7.parquet'
R = 6371007.181            # authalic radius, m
LEVEL_AREA_M2 = {6: 1.60e8, 7: 1.78e7, 9: 2.20e5, 10: 24408.0}

# label, file, level, native pixel size in degrees (None when projected), source geometry
INVENTORIES = [
    ('EDGAR v8.0', 'EDGAR/EDGAR_DGGS_methane_emissions_2020.csv', 6, (0.1, 0.1), 'global'),
    ('GFEI', 'GFEI_DGGS_methane_emissions_2016-2019-2020.csv', 6, (0.1, 0.1), 'global'),
    ('CAMS-REG v8.1', 'CAMS-REG_DGGS_methane_emissions_2005-2022.csv', 7, (0.1, 0.05),
     'europe'),
    ('U.S. Anthropogenic', 'US_DGGS_methane_emissions_2012-2018.csv', 6, (0.1, 0.1),
     'global'),
    ('U.S. Oil and Gas', 'US_OG_DGGS_methane_emissions_2021.csv', 6, (0.1, 0.1), 'global'),
    ('Canada', 'Canada_DGGS_methane_emissions_2018.csv', 6, (0.1, 0.1), 'global'),
    ('Mexico', 'Mexico_DGGS_methane_emissions_2015.csv', 6, (0.1, 0.1), 'global'),
    ('CMS Canada', 'CMS_Canada_DGGS_methane_emissions_2013.csv', 6, (0.1, 0.1), 'global'),
    ('CMS Mexico', 'CMS_Mexico_DGGS_methane_emissions_2010.csv', 6, (0.1, 0.1), 'global'),
    ('China coal mine (SACMS)', 'China_SACMS_DGGS_methane_emissions_2011.csv', 6,
     (0.25, 0.25), 'global'),
    ('India and Australia coal', 'India_Coal_DGGS_methane_emissions_2018.csv', 6,
     (0.1, 0.1), 'global'),
]

# Sources on a projected grid. An equal-area projection holds cell area constant; a
# conformal one does not, so those two are sampled over the domain outline. Outlines are
# used rather than the cell geometries, which run to gigabytes at levels 9 and 10.
PROJECTED = [
    ('China CHN-CH4', 6, None, None, 1.0e8),          # Albers equal-area conic
    ('Switzerland', 9, 'data/geojson/switzerland.geojson', ('EPSG:21781', 500.0), None),
    ('New York State', 10, 'data/geojson/newyorkstate.geojson', ('EPSG:32618', 100.0), None),
]


def projected_area_stats(outline_path, epsg, size, samples=160):
    """Mean and CV of the ground area of a constant-size projected pixel."""
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import Point

    outline = gpd.read_file(outline_path).unary_union
    x0, y0, x1, y1 = outline.bounds
    gx, gy = np.meshgrid(np.linspace(x0, x1, samples), np.linspace(y0, y1, samples))
    inside = np.array([outline.contains(Point(a, b))
                       for a, b in zip(gx.ravel(), gy.ravel())])
    lon, lat = gx.ravel()[inside], gy.ravel()[inside]

    fwd = Transformer.from_crs('EPSG:4326', epsg, always_xy=True)
    inv = Transformer.from_crs(epsg, 'EPSG:4326', always_xy=True)
    X, Y = fwd.transform(lon, lat)
    lons, lats = [], []
    for dx, dy in ((0, 0), (size, 0), (size, size), (0, size)):
        a, b = inv.transform(X + dx, Y + dy)
        lons.append(np.asarray(a))
        lats.append(np.asarray(b))
    lat0 = np.radians(lats[0])
    xs = [np.radians(v) * R * np.cos(lat0) for v in lons]
    ys = [np.radians(v) * R for v in lats]
    area = np.zeros_like(xs[0])
    for i in range(4):
        j = (i + 1) % 4
        area += xs[i] * ys[j] - xs[j] * ys[i]
    area = np.abs(area) / 2.0
    return float(area.mean()), float(area.std() / area.mean())


def pixel_area(lat_deg, dlon, dlat):
    """Area of one graticule cell centred on lat_deg, on the authalic sphere."""
    lo = np.radians(lat_deg - dlat / 2.0)
    hi = np.radians(lat_deg + dlat / 2.0)
    return R * R * np.radians(dlon) * (np.sin(hi) - np.sin(lo))


def load_centroids(kind):
    import geopandas as gpd
    if kind == 'europe':
        gdf = gpd.read_parquet(EUROPE_GEOM)
        pts = gdf.geometry.representative_point()
        return pd.DataFrame({'zoneID': gdf['zoneID'].astype(str), 'lat': pts.y})
    gdf = gpd.read_file(CENTROIDS)
    return pd.DataFrame({'zoneID': gdf['zoneID'].astype(str), 'lat': gdf.geometry.y})


def cell_ids(path):
    ids = []
    for chunk in pd.read_csv(DATA + path, usecols=['dggsID'], chunksize=2000000):
        ids.append(chunk['dggsID'].astype(str))
    return pd.unique(pd.concat(ids))


def rollup_residual(path):
    """Aggregate one level coarser by truncating the identifier, and compare totals."""
    total = 0.0
    parents = {}
    for chunk in pd.read_csv(DATA + path, chunksize=1000000):
        cols = [c for c in chunk.columns if c[:1].isdigit()]
        value = chunk[cols].sum(axis=1, skipna=True)
        total += float(value.sum())
        key = chunk['dggsID'].astype(str).str[:-1]
        for k, v in value.groupby(key).sum().items():
            parents[k] = parents.get(k, 0.0) + float(v)
    rolled = sum(parents.values())
    residual = abs(rolled - total) / total if total else float('nan')
    return len(parents), residual


def main():
    cache = {}
    rows = []
    for label, path, level, native, geom in INVENTORIES:
        if geom not in cache and geom != 'nys':
            cache[geom] = load_centroids(geom)
        cell_area = LEVEL_AREA_M2[level]

        cents = cache[geom]
        lats = cents[cents['zoneID'].isin(set(cell_ids(path)))]['lat'].to_numpy()
        areas = pixel_area(lats, *native)
        cv = float(areas.std() / areas.mean())
        ratio = cell_area / float(areas.mean())
        rows.append({'Inventory': label, 'Level': level,
                     'Native CV of cell area': cv,
                     'DGGS CV of cell area': 0.0,
                     'DGGS cell area / native pixel area': ratio})
        print('%-26s level %-3d native CV %7.4f  area ratio %s'
              % (label, level, cv, '%.2f' % ratio if ratio == ratio else 'n/a'))

    for label, level, outline, proj, fixed_area in PROJECTED:
        cell_area = LEVEL_AREA_M2[level]
        if proj is None:
            # equal-area projection, so cell area is constant by construction
            mean_area, cv = fixed_area, 0.0
        else:
            mean_area, cv = projected_area_stats(outline, *proj)
        rows.append({'Inventory': label, 'Level': level,
                     'Native CV of cell area': cv,
                     'DGGS CV of cell area': 0.0,
                     'DGGS cell area / native pixel area': cell_area / mean_area})
        print('%-26s level %-3d native CV %8.5f  area ratio %.2f'
              % (label, level, cv, cell_area / mean_area))

    print('\naggregation exactness, prefix roll-up one level coarser')
    for label, path in (('U.S. Anthropogenic', 'US_DGGS_methane_emissions_2012-2018.csv'),
                        ('Mexico', 'Mexico_DGGS_methane_emissions_2015.csv'),
                        ('Switzerland', 'Switzerland_DGGS_methane_emissions_2011.csv')):
        n, residual = rollup_residual(path)
        print('  %-22s %8d parent cells, relative residual %.3g' % (label, n, residual))

    pd.DataFrame(rows).to_csv('analysis_results/harmonization_criteria.csv', index=False)
    print('\nwrote analysis_results/harmonization_criteria.csv')


if __name__ == '__main__':
    main()
