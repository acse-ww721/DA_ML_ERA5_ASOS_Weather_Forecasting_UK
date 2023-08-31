import os
import pandas as pd
import numpy as np
import xarray as xr
from utils import folder_utils

# from asos_preprocessing import csv_to_nc4

# Example usage
country = "GB"
data_folder = "data"
data_read_category = "raw_data"
data_test_category = "test_data"
data_save_category = "processed_data"
output_folder = "ASOS_DATA"


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
