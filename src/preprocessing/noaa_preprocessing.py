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

    processed_df["TMP"] = raw_df["TMP"].apply(process_temperature)
    processed_df["DEW"] = raw_df["DEW"].apply(process_temperature)

    # Split "WIND"
    wind_df = raw_df["WND"].apply(process_wind)
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
