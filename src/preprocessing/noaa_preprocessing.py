import os
import re
import numpy as np
import xarray as xr
import pandas as pd
from utils import folder_utils
from tqdm import tqdm


def restore_decimal_format(encoded_value):
    # Convert data to decimal format
    if encoded_value in ["9999", "99999"]:
        return None  # Handle special cases if needed
    return float(encoded_value) / 10


def process_temperature(temperature_str):
    parts = temperature_str.split(",")
    symbol = parts[0][0]

    temperature = restore_decimal_format(int(parts[0][1:]))
    if symbol == "-":
        temperature = -temperature
    temperature_kelvin = temperature + 273.15
    return temperature_kelvin


def process_wind(wind_str):
    wind_parts = wind_str.split(",")
    wind_directory = wind_parts[0]
    wind_speed = restore_decimal_format(wind_parts[3])
    return pd.Series({"WIND_DIRECTORY": wind_directory, "WIND_SPEED": wind_speed})


def transform_data(df):
    # Create new columns based on the times
    times = [1, 3, 6, 12, 24]
    for t in times:
        df[f"{t}_hour_tp"] = np.nan  # Default value is NaN

    for col in ["AA1", "AA2", "AA3"]:
        # Check NaN
        mask = ~df[col].isna()
        filtered_data = df.loc[mask, col]
        filtered_data = filtered_data.dropna()  # Remove None & NaN

        # Split data
        if not filtered_data.empty:
            splits = filtered_data.str.split(",", expand=True)
            time_col = splits[0].astype(int)
            value_col = splits[1].apply(restore_decimal_format)

            for t in times:
                #
                idx_to_update = time_col[time_col == t].index
                df.loc[idx_to_update, f"{t}_hour_tp"] = value_col.loc[idx_to_update]

    return df


def noaa_data_preprocess(raw_df):
    # Reserve specified columns
    processed_df = raw_df[["STATION", "LATITUDE", "LONGITUDE", "ELEVATION"]]

    # Remove "UK"
    processed_df["NAME"] = raw_df["NAME"].str.replace(", UK", "").str.replace('"', "")

    # Split "DATE" to unified format
    date_time = pd.to_datetime(raw_df["DATE"])
    processed_df["DATE"] = date_time.dt.strftime("%Y%m%d")
    processed_df["TIME"] = date_time.dt.strftime("%H:%M")

    # Convert "TMP" and "DEW" from Celsius to Kelvin to match ERA5

    tqdm.pandas()
    processed_df["TMP"] = raw_df["TMP"].progress_apply(process_temperature)
    processed_df["DEW"] = raw_df["DEW"].progress_apply(process_temperature)

    # Split "WIND"
    wind_df = raw_df["WND"].progress_apply(process_wind)
    processed_df = pd.concat([processed_df, wind_df], axis=1)

    # Process AA1, AA2, AA3 as total precipitation
    # test_df[["AA1", "AA2", "AA3"]] = test_df[["AA1", "AA2", "AA3"]].astype(str)
    tp_df = transform_data(raw_df)[
        [
            "1_hour_tp",
            "3_hour_tp",
            "6_hour_tp",
            "12_hour_tp",
            "24_hour_tp",
        ]
    ].copy()
    processed_df = pd.concat([processed_df, tp_df], axis=1)

    # # filter the specific column
    # columns_to_keep = [
    #     "STATION",
    #     "NAME",
    #     "LATITUDE",
    #     "LONGITUDE",
    #     "ELEVATION",
    #     "DATE",
    #     "TIME",
    #     "1_hour_tp",
    #     "3_hour_tp",
    #     "6_hour_tp",
    #     "12_hour_tp",
    #     "24_hour_tp",
    #     "DEW",
    #     "TMP",
    #     "WIND_DIRECTORY",
    #     "WIND_SPEED",
    # ]
    #
    # # Save specific columns
    # processed_df = processed_df.filter(items=columns_to_keep)

    return processed_df


def save_noaa_processed_data(
    processed_df, country, data_folder, data_category, output_folder
):
    output_directory = folder_utils.create_folder(
        country, data_folder, data_category, output_folder
    )
    output_filename = f"{country}_NOAA__processed_data.nc"
    output_filepath = os.path.join(output_directory, output_filename)
    processed_df.to_csv(output_filepath, index=False, encoding="utf-8")
    print(f"{output_filename} done!")


def bulid_noaa_station_network(
    raw_df, country, data_folder, data_save_category, output_folder
):
    # build station_network
    # extract the specific columns
    selected_columns = ["STATION", "NAME", "LATITUDE", "LONGITUDE", "ELEVATION"]
    selected_df = raw_df[selected_columns]

    # Remove "UK"
    selected_df["NAME"] = (
        selected_df["NAME"].str.replace(", UK", "").str.replace('"', "")
    )

    # Drop duplicates
    unique_df = selected_df.drop_duplicates()

    # Remain 2 decimals
    unique_df["LATITUDE"] = unique_df["LATITUDE"].round(2)
    unique_df["LONGITUDE"] = unique_df["LONGITUDE"].round(2)
    unique_df["ELEVATION"] = unique_df["ELEVATION"].round(2)

    # save station_network

    folder_utils.create_folder(country, data_folder, data_save_category, output_folder)
    save_folder = folder_utils.find_folder(
        country, data_folder, data_save_category, output_folder
    )
    save_csv = "noaa_station_network.csv"
    save_path = os.path.join(save_folder, save_csv)
    unique_df.to_csv(save_path, index=False)

    # print(unique_df)


# Example usage

country = "GB"
data_folder = "data"
data_read_category = "raw_data"
data_save_category = "processed_data"
output_folder = "NOAA_DATA"

folder = folder_utils.find_folder(
    country, data_folder, data_read_category, output_folder
)
csv_name = "3419834.csv"
raw_csv_path = os.path.join(folder, csv_name)
raw_df = pd.read_csv(raw_csv_path)

processed_df = noaa_data_preprocess(raw_df)
save_noaa_processed_data(
    processed_df, country, data_folder, data_save_category, output_folder
)
