import os
import pandas as pd
import numpy as np
import xarray as xr
import gstools as gs
import geopandas as gpd
import matplotlib.pyplot as plt
from utils import folder_utils
from tqdm import tqdm
from asos_preprocessing import (
    get_year,
    get_asos_year_file_list,
    get_year_from_filename,
    filter_data,
)

# from asos_preprocessing import csv_to_nc4


def filter_data(df):
    """
    Filter data by deleting rows with missing values and wrong values
    """
    # Delete rows with missing values
    df = df.dropna()
    # Delete rows with wrong values
    df["time"] = pd.to_datetime(df["time"])  # Convert to datetime
    # If the time is not a whole hour, delete the row
    is_whole_hour = (df["time"].dt.minute == 0) & (df["time"].dt.second == 0)
    not_null = df["time"].notnull()
    latitude_condition = (df["latitude"] >= 50) & (df["latitude"] <= 58)
    longitude_condition = (df["longitude"] >= -6) & (df["longitude"] <= 2)
    # Combine all conditions
    combined_condition = (
        is_whole_hour & not_null & latitude_condition & longitude_condition
    )

    filtered_df = df[combined_condition]

    return filtered_df


def csv_to_nc4(
    merged_csv_path, year, country, data_folder, data_category, output_folder
):
    """
    Convert csv files to nc4 files by year
    """
    # Read csv files
    df = pd.read_csv(merged_csv_path)

    # Filter data
    df = filter_data(df)

    ds_in = xr.Dataset.from_dataframe(df.set_index(["latitude", "longitude", "time"]))
    ds_in = ds_in.sel(latitude=slice(50, 58), longitude=slice(-6, 2))
    ds_adjusted = ds_in.transpose("time", "latitude", "longitude")
    ds_adjusted["t2m"] = ds_adjusted["t2m"].astype("float32")

    # ddeg_out_lat = 0.25
    # ddeg_out_lon = 0.125
    # regridded_ds = regrid(
    #     ds_in, ddeg_out_lat, ddeg_out_lon, method="bilinear", reuse_weights=False
    # )

    # Save to nc4 file

    output_directory = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )
    output_filename = f"{country}_ASOS_filter_{year}.nc"
    output_path = os.path.join(output_directory, output_filename)
    ds_adjusted.to_netcdf(output_path)
    print(f"{output_filename} done!")


def krige_regrid(
    year_df_path, year, country, data_folder, data_category, output_folder
):
    # 1. Load the data
    df = pd.read_csv(year_df_path)
    df = filter_data(df)
    # lat = df["latitude"].values
    # lon = df["longitude"].values

    # 2. Create a new dataframe to store the interpolated data
    output_df = pd.DataFrame()

    # 3.Define the grid
    g_lon = np.linspace(-6.0, 1.875, 64)  # longitude
    g_lat = np.linspace(50.0, 57.75, 32)  # latitude
    # gridx, gridy = np.meshgrid(gridx, gridy)

    # 4. Drift term
    def north_south_drift(lat, lon):
        return lat

    # 4. Drift term
    def polynomial_drift(lat, lon):
        return [1, lat, lon, lat**2, lon**2, lat * lon]

    unique_times = df["time"].unique()

    # Iterate over each time
    for time_point in tqdm(unique_times):
        # 1. Load data for the specific time point
        time_df = df[df["time"] == time_point]
        t2m = time_df["t2m"].values
        lat = time_df["latitude"].values
        lon = time_df["longitude"].values

        # 2. Estimate the variogram
        bin_center, vario = gs.vario_estimate(
            (lat, lon), t2m, latlon=True, geo_scale=gs.KM_SCALE, max_dist=900
        )

        # 3. krige interpolation
        model = gs.Spherical(latlon=True, geo_scale=gs.KM_SCALE)
        model.fit_variogram(bin_center, vario, nugget=False)

        # 5. Universal Kriging

        uk = gs.krige.Universal(
            model=model,
            cond_pos=(lat, lon),
            cond_val=t2m,
            drift_functions=polynomial_drift,
        )

        uk.set_pos((g_lat, g_lon), mesh_type="structured")
        interpolated_values = uk(return_var=False)

        temp_df = pd.DataFrame(
            {
                "lat": np.tile(g_lat, len(g_lon)),
                "lon": np.repeat(g_lon, len(g_lat)),
                "time": [time_point] * len(g_lat) * len(g_lon),
                "t2m": interpolated_values.ravel(),
            }
        )

        output_df = pd.concat([output_df, temp_df], ignore_index=True)

    # Save to csv file
    output_directory = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )
    output_filename = f"{country}_ASOS_krige_{year}.csv"
    output_path = os.path.join(output_directory, output_filename)
    output_df.to_csv(output_path, index=False)

    # return output_df


########################################################################################

# Example usage
country = "GB"
data_folder = "data"
data_read_category = "raw_data"
data_test_category = "test_data"
data_save_category = "processed_data"
output_folder = "ASOS_DATA"

year_list = get_year(start_year=1979, end_year=2023)
csv_paths = get_asos_year_file_list(
    country, data_folder, data_save_category, output_folder
)
for year, csv_path in tqdm(zip(year_list, csv_paths)):
    krige_regrid(
        csv_path, year, country, data_folder, data_save_category, output_folder
    )
