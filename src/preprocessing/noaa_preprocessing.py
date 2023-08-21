import os
import re
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
    return pd.Series({"wind_directory": wind_directory, "wind_speed": wind_speed})


def process_aa(aa_str):
    if aa_str.strip() == "" or aa_str.strip().lower() == "nan":
        return None, None

    aa_parts = aa_str.split(",")
    hour = int(aa_parts[0])
    value = restore_decimal_format(aa_parts[1])
    return hour, value


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

    # Convert "AA1" column to strings
    raw_df["AA1"] = raw_df["AA1"].astype(str)

    aa_df = raw_df["AA1"].apply(process_aa)
    processed_df["6_hour_tp"] = aa_df.apply(lambda x: x[1] if x[0] == 6 else None)
    processed_df["12_hour_tp"] = aa_df.apply(lambda x: x[1] if x[0] == 12 else None)
    processed_df["24_hour_tp"] = aa_df.apply(lambda x: x[1] if x[0] == 24 else None)

    # filter the specific column
    columns_to_keep = [
        "STATION",
        "NAME",
        "LATITUDE",
        "LONGITUDE",
        "ELEVATION",
        "DATE",
        "TIME",
        "6_hour_tp",
        "12_hour_tp",
        "24_hour_tp",
        "DEW",
        "TMP",
        "WND",
    ]

    # 保留指定的列
    processed_df = processed_df.filter(items=columns_to_keep)

    return processed_df
