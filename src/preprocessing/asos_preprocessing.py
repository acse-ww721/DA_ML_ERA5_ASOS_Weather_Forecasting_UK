import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import dask.dataframe as dd
from tqdm import tqdm
from datetime import datetime, timedelta
from utils import folder_utils
from era5_preprocessing import regrid

"""V15"""


def get_csv_list(country, data_folder, data_category, output_folder):
    # Specify the folder path
    input_folder_path = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )
    # Initialize a dataframe to store all the data from the same country
    raw_df = pd.DataFrame()
    # Get the filenames of all CSV files under the folder except the station network file
    csv_files = [
        f
        for f in os.listdir(input_folder_path)
        if f.endswith(".csv") and "asos_station_network" not in f
    ]
    # Add path
    csv_file_paths = []  # Initialize a list to store the path of all csv files
    station_list = []  # Initialize a list to store the station name of all csv files
    for csv_file in tqdm(csv_files):
        csv_path = os.path.join(input_folder_path, csv_file)
        # Extract station from the filename
        station = csv_file.split("_")[1]
        csv_file_paths.append(csv_path)
        station_list.append(station)

    return csv_file_paths, station_list


def extract_data_to_df(country, data_folder, data_category, output_folder):
    # Specify the folder path
    input_folder_path = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )
    # Initialize a dataframe to store all the data from the same country
    raw_df = pd.DataFrame()
    # Get the filenames of all CSV files under the folder except the station network file
    csv_files = [
        f
        for f in os.listdir(input_folder_path)
        if f.endswith(".csv") and "asos_station_network" not in f
    ]
    # Read and merge the csv files in queue
    for csv_file in tqdm(csv_files):
        csv_file_path = os.path.join(input_folder_path, csv_file)
        df = pd.read_csv(csv_file_path)
        raw_df = pd.concat([raw_df, df], ignore_index=True)

    return raw_df


def time_rearrange(df):
    """
    For hourly initial times (such as 01:00:00, 02:00:00, etc.), hold.
    For the 20th minute of every hour (such as 01:20:00, 02:20:00, etc.),
    check whether the previous full point exists, and delete the 20-minute data if it exists,
    otherwise change to the previous full point.
    For the 50th minute of every hour (such as 01:50:00, 02:50:00, etc.),
    check whether the next full point exists, and if it exists, delete the 50-minute data,
    otherwise change to the next full point

    df is a DataFrame with a column named 'valid' containing time strings.
    Return the processed DataFrame.
    """
    # Create an assistant volume and convert time strings to datetime objects
    df["valid_datetime"] = pd.to_datetime(df["valid"], format="%Y-%m-%d %H:%M")

    # Delete the data before 1979-01-01
    cutoff_date = datetime(1979, 1, 1)

    # Check if the earliest date is after the cutoff date
    earliest_date = df["valid_datetime"].min()
    if earliest_date >= cutoff_date:
        print(
            f"The start date :  ({earliest_date.strftime('%d/%m/%Y')}) is after 19790101, terminate cutoff."
        )

    df = df[df["valid_datetime"] >= cutoff_date]

    # Create a list to store the indexes of rows to be deleted
    to_delete = []

    for index, row in tqdm(df.iterrows()):
        current_time = row["valid_datetime"]

        # If the minute is 20 minutes
        if current_time.minute == 20:
            prev_hour_time = current_time.replace(minute=0, second=0)
            if (
                prev_hour_time in df["valid_datetime"].values
            ):  # if the previous full point exists
                to_delete.append(index)  # record the current index for later deletion
            else:
                df.at[
                    index, "valid_datetime"
                ] = prev_hour_time  # otherwise change to the previous full point

        # if the minute is 50 minutes
        elif current_time.minute == 50:
            next_hour_time = (current_time + timedelta(hours=1)).replace(
                minute=0, second=0
            )
            if (
                next_hour_time in df["valid_datetime"].values
            ):  # if the next full point exists
                to_delete.append(index)  # record the current index for later deletion
            else:
                df.at[
                    index, "valid_datetime"
                ] = next_hour_time  # otherwise change to the next full point

    # delete the rows that need to be deleted
    df.drop(to_delete, inplace=True)

    # Delete the original time column
    df.drop(columns=["valid"], inplace=True)

    # Rename the new time column and  it save as iso 8601 format as datetime64[ns] automatically
    df.rename(columns={"valid_datetime": "time"}, inplace=True)
    # df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    return df


def process_asos_rawdata(raw_df):
    """
    Unified variable unit based on era5
    station: three or four character site identifier
    valid: observation time in UTC
    tmpc: Air Temperature in Celsius, typically @ 2 meters

    """

    # time preprocessing
    processed_df = time_rearrange(raw_df)

    # Convert Celsius to Kelvin for "tmpc" and rename the column
    processed_df.rename(columns={"tmpc": "t2m"}, inplace=True)
    processed_df["t2m"] = processed_df["t2m"] + 273.15
    processed_df["t2m"] = processed_df["t2m"].round(1)

    return processed_df


def save_asos_processed_data(
    processed_df, station, country, data_folder, data_category, output_folder
):
    output_directory = folder_utils.create_folder(
        country, data_folder, data_category, output_folder
    )
    output_filename = f"{country}_ASOS_{station}_processed_data.csv"
    output_filepath = os.path.join(output_directory, output_filename)
    processed_df.to_csv(output_filepath, index=False, encoding="utf-8")
    print(f"{output_filename} done!")




def merge_csv_station(country, data_folder, data_category, output_folder):
    """Merge all csv files in the folder and add station latlon information"""

    # Process station_network
    try:
        input_folder = folder_utils.find_folder(
            country, data_folder, data_category, output_folder
        )

        station_network_csv = "GB__asos_station_network.csv"
        station_network_csv_path = os.path.join(input_folder, station_network_csv)

        # Read station network CSV
        station_id_df = pd.read_csv(station_network_csv_path)

        columns_to_keep = [
            "ID",
            "Latitude",
            "Logitude",
        ]

        station_id_df = station_id_df[columns_to_keep]
        rename_mapping = {
            "ID": "station",
            "Latitude": "latitude",
            "Logitude": "longitude",  # Fixed the typo
        }

        station_id_df.rename(columns=rename_mapping, inplace=True)

        # Check if input folder is empty
        files_in_directory = os.listdir(input_folder)
        if not files_in_directory:
            print("Error: The specified folder is empty.")
            return None

        # Iterate through CSV files in the folder
        merged_df_list = []
        for filename in tqdm(files_in_directory):
            if filename.startswith("GB_ASOS_") and filename.endswith("_processed_data.csv"):
                csv_path = os.path.join(input_folder, filename)

                # Use chunk reading for large files
                chunk_size = 10000  # Adjust as needed
                chunks = pd.read_csv(csv_path, chunksize=chunk_size)
                for chunk in chunks:
                    try:
                        merged_df = pd.merge(station_id_df, chunk, on="station", how="left")
                        merged_df_list.append(merged_df)
                    except Exception as e:
                        print(f"Error processing chunk in file {filename}: {e}")
                del chunk

        # Concatenate all dataframes in the list
        merged_df_all = pd.concat(merged_df_list, ignore_index=True)
        desired_order = ["latitude", "longitude", "time", "t2m"]
        merged_df_all = merged_df_all[desired_order]
        # Drop duplicates based on time, latitude, and longitude
        merged_df_all = merged_df_all.drop_duplicates(subset=['time', 'latitude', 'longitude'])

        # After merging all the dataframes and dropping duplicates:
        # Convert 'time' column to datetime type
        merged_df_all['time'] = pd.to_datetime(merged_df_all['time']) # re-ensure the dtype

        # Group by year
        grouped = merged_df_all.groupby(merged_df_all['time'].dt.year)

        # Save each group as a CSV
        for year, group in tqdm(grouped):
            output_filename = f"{country}_merged_ASOS_{year}.csv"
            output_path = os.path.join(output_folder, output_filename)
            group.to_csv(output_path, index=False)
            print(f"{output_filename} saved!")

        del merged_df_list  # Further release memory

    except Exception as e:
        print(f"Error processing files: {e}")
        return None

def save_asos_merged_data(
    merged_df, country, data_folder, data_category, output_folder
):
    output_directory = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )
    output_filename = f"{country}_ASOS_merged_data.csv"
    output_filepath = os.path.join(output_directory, output_filename)
    merged_df.to_csv(output_filepath, index=False, encoding="utf-8")
    print(f"{output_filename} done!")
    return output_filepath

# def custom_date_parser(x):
#     if pd.isna(x):
#         return pd.NaT
#     return pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')

def csv_to_nc4(merged_csv_path, country, data_folder, data_category, output_folder):
    """Convert the merged CSV file to netCDF4 format by year."""
    try:
        # Function to process each partition to xarray Dataset
        def process_partition_to_xarray(df_partition):
            data_vars = {
                't2m': df_partition['t2m'].values
            }
            coords = {
                'latitude': df_partition['latitude'].values,
                'longitude': df_partition['longitude'].values,
                'time': df_partition['time'].values
            }
            ds = xr.Dataset(data_vars, coords=coords)
            return ds

        # 1. Use Dask's lazy computation strategy.
        chunksize = 200_000
        dtype_optimization = {
            't2m': 'float32',
            'latitude': 'float64',
            'longitude': 'float64',
        }

        merged_dask_df_iter = dd.read_csv(
            merged_csv_path,
            blocksize=chunksize,
            dtype=dtype_optimization,
            parse_dates=['time'],
            date_format='%Y-%m-%d %H:%M:%S'
        )

        output_directory = folder_utils.find_folder(
            country, data_folder, data_category, output_folder
        )

        meta = xr.DataArray(np.array([[[0.]]]),
                            coords={'latitude': [0.], 'longitude': [0.], 'time': [pd.Timestamp('2000-01-01')]},
                            dims=['latitude', 'longitude', 'time'],
                            name='t2m')

        # Convert Dask DataFrame partitions to xarray and compute the result
        ds_list = merged_dask_df_iter.map_partitions(process_partition_to_xarray,meta=meta).compute().tolist()

        # Combine chunks into one large dataset
        combined_ds = xr.concat(ds_list, dim='index')

        # Further processing
        combined_ds = combined_ds.sel(latitude=slice(58, 50), longitude=slice(-6, 2))
        ddeg_out_lat = 0.25
        ddeg_out_lon = 0.125
        # regridded_ds = regrid(combined_ds, ddeg_out_lat, ddeg_out_lon, method="bilinear", reuse_weights=False)
        years = combined_ds["time.year"].unique().values

        with tqdm(years) as t_years:
            for year in t_years:
                year_ds = combined_ds.sel(time=str(year))
                output_filename_nc = f"{country}_ASOS_bf_regrid_data_{year}.nc"
                output_filepath = os.path.join(output_directory, output_filename_nc)
                year_ds.to_netcdf(output_filepath)
                print(f"{output_filename_nc} saved !")

        # Memory cleanup
        del combined_ds
        gc.collect()

        return True

    except Exception as e:
        print(f"Error processing and saving data: {e}")
        return False



# Example usage

country = "GB"
data_folder = "data"
data_read_category = "raw_data"
data_test_category = "test_data"
data_save_category = "processed_data"
output_folder = "ASOS_DATA"

################ Process ASOS raw data ################
csv_list, station_list = get_csv_list(
    country, data_folder, data_read_category, output_folder
)

for csv_path, station in tqdm(zip(csv_list, station_list)):
    print(csv_path)
    try:
        raw_df = pd.read_csv(csv_path)
        processed_df = process_asos_rawdata(raw_df)
        save_asos_processed_data(
            processed_df,
            station,
            country,
            data_folder,
            data_save_category,
            output_folder,
        )
    except Exception as e:
        print(f"An error occurred for {csv_path}: {e}")
        continue  # Continue to the next iteration

############### Make the same format as era5 dataset including cutoff and regrid ##################

# Merge all csv files in the folder and add station latlon information
merged_df = merge_csv_station(country, data_folder, data_save_category, output_folder)
merged_csv_path = save_asos_merged_data(merged_df,country,data_folder,data_save_category, output_folder)
csv_to_nc4(merged_csv_path, country, data_folder, data_save_category, output_folder)
