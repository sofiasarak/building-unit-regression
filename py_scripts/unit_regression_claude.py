import pandas as pd
from shapely.geometry import box
import numpy as np
import geopandas as gpd
import dask_geopandas as dgpd
import os
import logging
import statsmodels.formula.api as smf

os.environ['PROJ_LIB'] = '/opt/anaconda3/share/proj'

NPARTITIONS = 8  # tune down to 4 if hitting memory pressure, up to 16 if you have cores + RAM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("/capstone/electrigrid/outputs/electrigrid_test_pipeline.log"),
        logging.StreamHandler(),  # still prints to console
    ]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def parallel_sjoin(left_gdf, right_gdf, npartitions=NPARTITIONS, **kwargs):
    """Partition left GDF and sjoin in parallel; right is broadcast to all workers."""
    left_dask = dgpd.from_geopandas(left_gdf, npartitions=npartitions)
    return left_dask.sjoin(right_gdf, **kwargs).compute()


# ---------------------------------------------------------------------------
# ONE-TIME CONVERSION  (comment out after first run)
# ---------------------------------------------------------------------------

def convert_to_parquet():
    """Run once to convert slow GDB/GPKG files to fast Parquet."""
    log.info("Converting parcels GDB -> parquet...")
    parcels = gpd.read_file(
        "data/Parcels_CA_2014.gdb",
        layer="CA_PARCELS_STATEWIDE"
    ).to_crs(epsg=4326)
    parcels.to_parquet("data/parcels_ca.parquet")

    log.info("Converting zillow GPKG -> parquet...")
    zillow = gpd.read_file("data/final_zillow.gpkg").to_crs(epsg=4326)
    zillow.to_parquet("data/zillow.parquet")

    log.info("Conversion complete. Comment out convert_to_parquet() for future runs.")


convert_to_parquet()  # <-- uncomment for first run only


# ---------------------------------------------------------------------------
# LOAD DATA  (fast parquet reads)
# ---------------------------------------------------------------------------

log.info("Loading data...")
parcels  = dgpd.read_parquet("data/parcels_ca.parquet").compute().to_crs(epsg=4326)
zillow   = dgpd.read_parquet("data/zillow.parquet").compute().to_crs(epsg=4326)
building = dgpd.read_parquet(
    "../../../../../capstone/electrigrid/data/buildings/buildings_ca.parquet"
).compute().to_crs(epsg=4326)


# ---------------------------------------------------------------------------
# FILTER ZILLOW
# ---------------------------------------------------------------------------

zillow_multi  = zillow[(zillow['type'] == "Multi") & (zillow['code'] != "RR106")]
zillow_single = zillow[(zillow['type'] == "Single") | (zillow['code'] == "RR106")]


# ---------------------------------------------------------------------------
# FIND MULTI-FAMILY BUILDINGS  (Zillow -> Parcel -> Buildings)
# ---------------------------------------------------------------------------

log.info("Filtering residential parcels...")
valid_parcels = parallel_sjoin(
    parcels, zillow_multi,
    how="inner", predicate="intersects"
)[parcels.columns].index.unique()
parcels_res = parcels.loc[valid_parcels]

# assert len(parcels_res) < len(parcels)

log.info("Filtering residential buildings...")
valid_buildings = parallel_sjoin(
    building, parcels_res,
    predicate="intersects"
).index.unique()
buildings_res = building.loc[valid_buildings]

# assert len(buildings_res) < len(building)


# ---------------------------------------------------------------------------
# CALCULATE VOLUME  (multi + single/condo)
# ---------------------------------------------------------------------------

log.info("Joining buildings to zillow (multi)...")
building_zillow_multi = gpd.sjoin(
    buildings_res, zillow_multi,
    how="left", predicate="intersects"
)

log.info("Joining buildings to zillow (single/condo)...")
building_zillow_single = gpd.sjoin(
    building, zillow_single,
    how="inner", predicate="intersects"
)

building_zillow_all = pd.concat([building_zillow_multi, building_zillow_single])

building_m = building_zillow_all.to_crs("EPSG:6933")
building_m['area_m2']   = building_m.geometry.area
building_m.rename(columns={"height": "height_m"}, inplace=True)
building_m['volume_m3'] = building_m['area_m2'] * building_m['height_m']

non_multi  = building_m[(building_m['type'] == "Single") | (building_m['code'] == "RR106")]
building_m = building_m[(building_m['type'] == "Multi") & (building_m['code'] != "RR106")]


# ---------------------------------------------------------------------------
# REGRESSION TO PREDICT MISSING UNIT COUNTS
# ---------------------------------------------------------------------------

building_w_units = building_m[~building_m['unit'].isna()]
assert building_w_units['unit'].isna().sum() == 0

results = smf.ols('unit ~ volume_m3', data=building_w_units).fit()
building_w_units = building_w_units.copy()
building_w_units['residual'] = results.resid

std = building_w_units['residual'].std()
building_units_clean = building_w_units[building_w_units['residual'].abs() <= 2 * std]
building_outliers    = building_w_units[building_w_units['residual'].abs() >  2 * std]

results_clean = smf.ols('unit ~ volume_m3', data=building_units_clean).fit()
intercept, slope = results_clean.params[0], results_clean.params[1]

missing_units = building_m[building_m['unit'].isna()]
missing_outlier_units = pd.concat([building_outliers, missing_units])
assert len(missing_units) < len(missing_outlier_units)

missing_outlier_units_pred = (
    missing_outlier_units
    .drop('unit', axis=1)
    .reset_index(drop=True)
)
missing_outlier_units_pred['unit'] = round(
    intercept + missing_outlier_units_pred['volume_m3'] * slope
)

multi_complete = (
    pd.concat([building_units_clean, missing_outlier_units_pred])
    .drop(columns=['residual'], errors='ignore')
    .drop(columns=['index_right'], errors='ignore')
    .to_crs(zillow.crs)
)


# ---------------------------------------------------------------------------
# AGGREGATE UNITS BY PARCEL
# ---------------------------------------------------------------------------

log.info("Aggregating units by parcel...")
multi_by_parcel = gpd.sjoin(
    parcels_res, multi_complete,
    predicate="intersects"
)
assert len(multi_by_parcel) < len(multi_complete)

# reuse multi_by_parcel — no redundant join
summed_units = multi_by_parcel.groupby("PARNO")['unit'].sum()
assert len(summed_units) < len(multi_by_parcel)

multi_summed_units = parcels_res.join(summed_units).dropna(subset=['unit'])
assert len(multi_summed_units) < len(multi_by_parcel)


# ---------------------------------------------------------------------------
# NON-MULTI (SINGLE + CONDO)
# ---------------------------------------------------------------------------

non_multi = non_multi[
    ['type', 'source', 'id', 'height_m', 'var', 'region', 'bbox', 'geometry', 'area_m2', 'volume_m3']
].to_crs(zillow.crs)

log.info("Joining zillow to non-multi buildings...")
non_multi_points = gpd.sjoin(
    zillow, non_multi,
    how="inner", predicate="intersects"
)


# ---------------------------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------------------------

log.info("Saving outputs...")
multi_summed_units.to_file("data/multi_summed_units_ca.geojson", driver='GeoJSON')
non_multi_points.to_file("data/non_multi_points_ca.geojson",    driver='GeoJSON')

log.info("Done.")
