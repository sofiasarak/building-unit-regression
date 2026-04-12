# import necessary libraries
import pandas as pd
from shapely.geometry import box
import numpy as np
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import fiona
import pandas as pd

# statistical libraries
#import sys
#!{sys.executable} -m pip install statsmodels
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

os.environ['PROJ_LIB'] = '/opt/anaconda3/share/proj'

# load for small section of PGE counties
# load parcels only for PGE counties
parcels = gpd.read_file(
    "data/Parcels_CA_2014.gdb",
    layer="CA_PARCELS_STATEWIDE").to_crs(epsg=4326)

# import Zillow data (make take 10-20 minutes)
fp = os.path.join('data', 'final_zillow.gpkg')
zillow = gpd.read_file(fp).to_crs(epsg=4326)

# import building footprint as geopandas dataframe (may take 1-5 minutes)
fp = "../../../../../capstone/electrigrid/data/buildings/buildings_ca.parquet"
building = gpd.read_parquet(fp).to_crs(epsg=4326)

## FINDING ALL MULTI-FAMILY BUILDINGS BY JOINING ZILLOW -> PARCEL, THEN PARCEL -> BUILDINGS

# select only multi-family data from Zillow
zillow_multi = zillow[zillow['type'] == "Multi"]
zillow_multi = zillow_multi[zillow_multi['code'] != "RR106"]

## crop only to residential parcels
# keep the indices where multi-family homes match to parcels (.index.unique() de-duplicates)
valid_parcels = parcels.sjoin(zillow_multi, how = "inner", predicate="intersects")[parcels.columns].index.unique()

# select the parcels that match these indices
parcels_res = parcels.loc[valid_parcels]

# confirm that joining with Zillow decreases the number of parcels
assert len(parcels_res) < len(parcels)

# crop to residential buildings (by keeping only those within residential parcels)
valid_buildings = building.sjoin(parcels_res, predicate="intersects").index.unique()
buildings_res = building.loc[valid_buildings]

# confirm that joining with Zillow decreased the number of buildings
assert len(buildings_res) < len(building)

## join parcels to buildings (keeping observations as parcels, but with building attributes)
# sum number of units per parcel
#units_sum = parcels_res.sjoin(zillow_multi, predicate="intersects").groupby(level=0)["unit"].sum()

# join on parcels with summed number of units
#parcels_res = parcels_res.join(units_sum)



## CALCULATING VOLUME FOR BOTH SINGLE AND MULTI-FAMILY HOMES

# keep all residential buildings, and add zillow points only where they match up (MULTI)
building_zillow_multi = gpd.sjoin(
    buildings_res,
    zillow_multi,
    how = "left",
    predicate = "intersects")

# filter for only single and condo units
zillow_single = zillow[(zillow['type'] == "Single") | (zillow['code'] == "RR106")]

# keep all residential buildings, and add zillow points only where they match up (SINGLE & CONDOS)
building_zillow_single = gpd.sjoin(
    building,
    zillow_single,
    how = "inner",
    predicate = "intersects")

# combine all for volume calculation
building_zillow_all = pd.concat([building_zillow_multi, building_zillow_single])

# reproject data frame to crs with meters as units
building_m = building_zillow_all.to_crs("EPSG:6933")

# create column from polygon area
building_m['area_m2'] = building_m.geometry.area

# rename height column to be clear about units
building_m.rename(columns={"height":"height_m"}, inplace = True)

# create volume column
building_m['volume_m3'] = building_m['area_m2'] * building_m['height_m']

# save single and condos as its own df
non_multi = building_m[(building_m['type'] == "Single") | (building_m['code'] == "RR106")]

# now that single unit homes also have volume data, we can drop them
building_m = building_m[building_m['type'] == "Multi"]
building_m = building_m[building_m['code'] != "RR106"]



## BUILDING REGRESSION TO PREDICT UNIT DATA WHERE IT IS MISSING

# keep only observations with unit data
building_w_units = building_m[~building_m['unit'].isna()]

assert building_w_units['unit'].isna().sum() == 0

# run model
results = smf.ols('unit ~ volume_m3', data=building_w_units).fit()

# add residuals as a column
building_w_units['residual'] = results.resid.copy()

# keep only observations that are less/equal to 2 standard deviations from residuals
building_units_clean = building_w_units[building_w_units['residual'].abs() <= 2 * building_w_units['residual'].std()]

# save outliers, as we will re-predict them using the regression
building_outliers = building_w_units[building_w_units['residual'].abs() > 2 * building_w_units['residual'].std()]

# rerun linear regression
results_clean = smf.ols('unit ~ volume_m3', data=building_units_clean).fit()

# save variables
intercept = results_clean.params[0]
slope = results_clean.params[1]

# extract just the multi-family homes where unit info is missing
missing_units = building_m[building_m['unit'].isna()]

# combine dataframes with missing unit data as well as outliers (since both will be predicted)
missing_outlier_units = pd.concat([building_outliers, missing_units])

assert len(missing_units) < len(missing_outlier_units)

# replace unit column with prediction
missing_outlier_units_pred = missing_outlier_units.copy().drop('unit', axis = 1)

missing_outlier_units_pred = missing_outlier_units_pred.reset_index(drop=True)

missing_outlier_units_pred['unit'] = round(intercept + missing_outlier_units_pred['volume_m3'] * slope)

# combine multi-family homes data frames
multi_complete = pd.concat([building_w_units, missing_outlier_units_pred]).to_crs(zillow.crs)

# drop excess columns
multi_complete = multi_complete.drop(['residual'], axis = 1)
multi_complete = multi_complete.drop(['index_right'], axis = 1) 

## UNIT PREDICTION COMPLETE



## AGGREGATE BY PARCEL (SUM MULTI-HOME UNITS)

# find the parcels that contain multi-family homes
multi_by_parcel = parcels_res.sjoin(multi_complete, predicate="intersects")

# when aggregated to parcel, there should be less multi-family home observations
assert len(multi_by_parcel) < len(multi_complete)

# join complete multi-family homes data frame to parcels, then sum units by parcel
summed_units = parcels_res.sjoin(multi_complete, predicate="intersects").groupby(["PARNO"])['unit'].sum()

assert len(summed_units) < len(multi_by_parcel) # this should hold because multi_by_parcel has rows for each building, 
                                                # whereas multi_summed_units is by parcel (and there are less parcels than buildings)

# join unit sums to the parcel geometries themselves, keeping only where units were summed
multi_summed_units = parcels_res.join(summed_units).dropna(subset = ['unit'])

assert len(multi_summed_units) < len(multi_by_parcel)

## SAVING NON-MULTI-FAMILY (SINGLE AND CONDO) OBSERVATIONS

# keep only variables of interest
non_multi = non_multi[['type', 'source', 'id', 'height_m', 'var', 'region', 'bbox', 'geometry', 'area_m2', 'volume_m3']].to_crs(zillow.crs)

# join Zillow data to non-multi family homes (takes ~1 minute)
non_multi_points = gpd.sjoin(
    zillow, # left df's geometry is always kept
    non_multi,
    how = "inner",
    predicate = "intersects")

# save data frame copies (takes >10 min)
multi_summed_units_ca = multi_summed_units.copy()
non_multi_points_ca = non_multi_points.copy()

# save
multi_summed_units_ca.to_file("data/multi_summed_units_ca.geojson", driver='GeoJSON')
non_multi_points_ca.to_file("data/non_multi_points_ca.geojson", driver='GeoJSON')

