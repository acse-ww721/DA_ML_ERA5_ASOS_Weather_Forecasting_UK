# Name: Wenqi Wang
# Github username: acse-ww721

import os
import pandas as pd
import xarray as xr
from tqdm import tqdm
from datetime import datetime, timedelta
from utils import folder_utils

"""V15"""


def get_csv_list(country, data_folder, data_category, output_folder):
    """
    Get a list of CSV files and their corresponding station names for a specific country.

    This function searches for CSV files in the specified folder and returns a list of file paths along with their corresponding station names.

    Args:
        country (str): The country code or identifier.
        data_folder (str): The path to the data folder.
        data_category (str): The data category.
        output_folder (str): The output folder name.

    Returns:
        tuple: A tuple containing two lists - the list of CSV file paths and the list of corresponding station names.

    Example:
        >>> country = "GB"
        >>> data_folder = "data"
        >>> data_category = "processed_data"
        >>> output_folder = "ASOS_DATA"
        >>> csv_file_paths, station_list = get_csv_list(country, data_folder, data_category, output_folder)
        # Retrieves a list of CSV files and their corresponding station names for the specified country.
    """
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
    """
    Extract and merge data from multiple CSV files into a single DataFrame for a specific country.

    This function searches for CSV files in the specified folder, reads each file, and merges them into a single DataFrame.

    Args:
        country (str): The country code or identifier.
        data_folder (str): The path to the data folder.
        data_category (str): The data category.
        output_folder (str): The output folder name.

    Returns:
        pandas.DataFrame: A DataFrame containing merged data from multiple CSV files.

    Example:
        >>> country = "GB"
        >>> data_folder = "data"
        >>> data_category = "processed_data"
        >>> output_folder = "ASOS_DATA"
        >>> merged_data_df = extract_data_to_df(country, data_folder, data_category, output_folder)
        # Extracts and merges data from multiple CSV files into a single DataFrame for the specified country.
    """
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
    Args:
        df (pandas.DataFrame): A DataFrame containing a 'valid' column with time strings.

    Returns:
        pandas.DataFrame: The processed DataFrame with rearranged time values.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'valid': ['2023-09-01 01:20:00', '2023-09-01 02:50:00', '2023-09-01 03:00:00']})
        >>> rearranged_df = time_rearrange(df)
        # Rearranges the time values in the DataFrame based on the specified rules.
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
    Process raw ASOS weather data to unify variable units based on ERA5 format.

    Args:
        raw_df (pandas.DataFrame): Raw ASOS weather data DataFrame.

    Returns:
        pandas.DataFrame: Processed DataFrame with unified variable units.

    Example:
        >>> import pandas as pd
        >>> raw_df = pd.DataFrame({'station': ['ABC', 'XYZ'],
        ...                        'valid': ['2023-09-01 01:20:00', '2023-09-01 02:50:00'],
        ...                        'tmpc': [20.5, 25.0]})
        >>> processed_df = process_asos_rawdata(raw_df)
        # Processes raw ASOS weather data to unify variable units and convert Celsius to Kelvin.
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
    """
    Save processed ASOS weather data to a CSV file.

    Args:
        processed_df (pandas.DataFrame): Processed ASOS weather data DataFrame.
        station (str): Station identifier.
        country (str): Country name.
        data_folder (str): Data folder name.
        data_category (str): Data category name.
        output_folder (str): Output folder name.

    Example:
        >>> import pandas as pd
        >>> processed_df = pd.DataFrame({'station': ['ABC', 'XYZ'],
        ...                              'time': ['2023-09-01T01:20:00', '2023-09-01T02:50:00'],
        ...                              't2m': [293.65, 298.15]})
        >>> save_asos_processed_data(processed_df, 'ABC', 'USA', 'data', 'weather', 'output')
        # Saves processed ASOS weather data to a CSV file.
    """
    output_directory = folder_utils.create_folder(
        country, data_folder, data_category, output_folder
    )
    output_filename = f"{country}_ASOS_{station}_processed_data.csv"
    output_filepath = os.path.join(output_directory, output_filename)
    processed_df.to_csv(output_filepath, index=False, encoding="utf-8")
    print(f"{output_filename} done!")


def merge_csv_station(country, data_folder, data_category, output_folder):
    """
    Merge all CSV files in the folder and add station latlon information.
    The merged CSV files are saved by year.

    Args:
        country (str): Country name.
        data_folder (str): Data folder name.
        data_category (str): Data category name.
        output_folder (str): Output folder name.

    Example:
        >>> merge_csv_station('GB', 'data', 'processed_data', 'ERA5_DATA')
        # Merges CSV files in the specified folder, adds station latlon information, and saves the merged files by year.
    """

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
            if filename.startswith("GB_ASOS_") and filename.endswith(
                "_processed_data.csv"
            ):
                csv_path = os.path.join(input_folder, filename)

                # Use chunk reading for large files
                chunk_size = 10000  # Adjust as needed
                chunks = pd.read_csv(csv_path, chunksize=chunk_size)
                for chunk in chunks:
                    try:
                        merged_df = pd.merge(
                            station_id_df, chunk, on="station", how="left"
                        )
                        merged_df_list.append(merged_df)
                    except Exception as e:
                        print(f"Error processing chunk in file {filename}: {e}")
                del chunk

        # Concatenate all dataframes in the list
        merged_df_all = pd.concat(merged_df_list, ignore_index=True)
        desired_order = ["latitude", "longitude", "time", "t2m"]
        merged_df_all = merged_df_all[desired_order]
        # Drop duplicates based on time, latitude, and longitude
        merged_df_all = merged_df_all.drop_duplicates(
            subset=["time", "latitude", "longitude"]
        )

        # After merging all the dataframes and dropping duplicates:
        # Convert 'time' column to datetime type
        merged_df_all["time"] = pd.to_datetime(
            merged_df_all["time"]
        )  # re-ensure the dtype

        # Group by year
        grouped = merged_df_all.groupby(merged_df_all["time"].dt.year)

        # Save each group as a CSV
        output_directory = folder_utils.find_folder(
            country, data_folder, data_category, output_folder
        )
        year_list = []
        for year, group in tqdm(grouped):
            output_filename = f"{country}_merged_ASOS_{year}.csv"
            output_path = os.path.join(output_directory, output_filename)
            group.to_csv(output_path, index=False, encoding="utf-8")
            year_list.append(year)
            print(f"{output_path} saved!")

        del merged_df_list  # Further release memory

    except Exception as e:
        print(f"Error processing files: {e}")
        return None


def get_year(start_year, end_year):
    # start_year = 1979
    # end_year = 2023
    year_list = list(range(start_year, end_year + 1))

    # Convert the integer list to a string list
    year_str_list = [str(year) for year in year_list]

    return year_str_list


def get_year_from_filename(filename):
    # extract year from filename
    # filename = "GB_merged_ASOS_1979.csv"
    parts = filename.split("_")
    return parts[3]


def get_asos_year_file_list(country, data_folder, data_category, output_folder):
    input_folder_path = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )
    csv_files = [
        f
        for f in os.listdir(input_folder_path)
        if f.endswith(".csv") and "_merged_ASOS_" in f
    ]

    # sort by year
    csv_files.sort(key=lambda x: get_year_from_filename(x))

    return [
        os.path.join(input_folder_path, csv_file) for csv_file in csv_files
    ]  # return the full path


# def csv_to_nc4(
#     merged_csv_path, year, country, data_folder, data_category, output_folder
# ):
#     """Convert the merged CSV file to netCDF4 format by year."""
#     parse_dates = ["time"]
#     dtype_optimization = {
#         "t2m": "float32",
#         "latitude": "float64",
#         "longitude": "float64",
#     }
#     df = pd.read_csv(merged_csv_path, dtype=dtype_optimization, parse_dates=parse_dates)
#     ds_in = xr.Dataset.from_dataframe(df.set_index(["latitude", "longitude", "time"]))
#     ds_in = ds_in.sel(
#         latitude=slice(50, 58), longitude=slice(-6, 2)
#     )  # not reverse as era5
#     ddeg_out_lat = 0.25
#     ddeg_out_lon = 0.125
#     regridded_ds = regrid(
#         ds_in, ddeg_out_lat, ddeg_out_lon, method="bilinear", reuse_weights=False
#     )
#
#     output_directory = folder_utils.find_folder(
#         country, data_folder, data_category, output_folder
#     )
#     output_filename = f"{country}_ASOS_regird_{year}.nc"
#     output_path = os.path.join(output_directory, output_filename)
#     regridded_ds.to_netcdf(output_path)
#     print(f"{output_filename} done!")


# def csv_to_nc4(merged_csv_path, country, data_folder, data_category, output_folder):
#     """Convert the merged CSV file to netCDF4 format by year."""
#     try:
#         # Function to process each partition to xarray Dataset
#         def process_partition_to_xarray(df_partition):
#             data_vars = {
#                 't2m': df_partition['t2m'].values
#             }
#             coords = {
#                 'latitude': df_partition['latitude'].values,
#                 'longitude': df_partition['longitude'].values,
#                 'time': df_partition['time'].values
#             }
#             ds = xr.Dataset(data_vars, coords=coords)
#             return ds
#
#         # 1. Use Dask's lazy computation strategy.
#         chunksize = 200_000
#         dtype_optimization = {
#             't2m': 'float32',
#             'latitude': 'float64',
#             'longitude': 'float64',
#         }
#
#         merged_dask_df_iter = dd.read_csv(
#             merged_csv_path,
#             blocksize=chunksize,
#             dtype=dtype_optimization,
#             parse_dates=['time'],
#             date_format='%Y-%m-%d %H:%M:%S'
#         )
#
#         output_directory = folder_utils.find_folder(
#             country, data_folder, data_category, output_folder
#         )
#
#         meta = xr.DataArray(np.array([[[0.]]]),
#                             coords={'latitude': [0.], 'longitude': [0.], 'time': [pd.Timestamp('2000-01-01')]},
#                             dims=['latitude', 'longitude', 'time'],
#                             name='t2m')
#
#         # Convert Dask DataFrame partitions to xarray and compute the result
#         ds_list = merged_dask_df_iter.map_partitions(process_partition_to_xarray,meta=meta).compute().tolist()
#
#         # Combine chunks into one large dataset
#         combined_ds = xr.concat(ds_list, dim='index')
#
#         # Further processing
#         combined_ds = combined_ds.sel(latitude=slice(58, 50), longitude=slice(-6, 2))
#         ddeg_out_lat = 0.25
#         ddeg_out_lon = 0.125
#         # regridded_ds = regrid(combined_ds, ddeg_out_lat, ddeg_out_lon, method="bilinear", reuse_weights=False)
#         years = combined_ds["time.year"].unique().values
#
#         with tqdm(years) as t_years:
#             for year in t_years:
#                 year_ds = combined_ds.sel(time=str(year))
#                 output_filename_nc = f"{country}_ASOS_bf_regrid_data_{year}.nc"
#                 output_filepath = os.path.join(output_directory, output_filename_nc)
#                 year_ds.to_netcdf(output_filepath)
#                 print(f"{output_filename_nc} saved !")
#
#         # Memory cleanup
#         del combined_ds
#         gc.collect()
#
#         return True
#
#     except Exception as e:
#         print(f"Error processing and saving data: {e}")
#         return False


def filter_data(df):
    """
    Filter data by deleting rows with missing values and incorrect values.

    Args:
        df (pd.DataFrame): Input DataFrame containing time, latitude, and longitude columns.

    Returns:
        pd.DataFrame: Filtered DataFrame with valid rows.

    Example:
        >>> filtered_df = filter_data(raw_data_df)
        # Filters the DataFrame to remove rows with missing values and rows that don't meet specific conditions.
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
    Convert CSV files to NetCDF4 (nc4) files by year.

    Args:
        merged_csv_path (str): Path to the merged CSV file.
        year (int): The year for which the data is being converted.
        country (str): Country code or name.
        data_folder (str): Folder where the data is stored.
        data_category (str): Category of the data.
        output_folder (str): Folder where the NetCDF4 files will be saved.

    Returns:
        None

    Example:
        >>> csv_to_nc4("merged_data.csv", 2022, "UK", "data_folder", "data_category", "output_folder")
        # Converts the CSV data to a NetCDF4 file for the specified year.
    """
    # Read csv files
    df = pd.read_csv(merged_csv_path)

    # Filter data
    df = filter_data(df)

    ds_in = xr.Dataset.from_dataframe(df.set_index(["latitude", "longitude", "time"]))
    ds_in = ds_in.sel(latitude=slice(50, 58), longitude=slice(-6, 2))
    ds_adjusted = ds_in.transpose("time", "latitude", "longitude")
    ds_adjusted["t2m"] = ds_adjusted["t2m"].astype("float32")

    # Save to nc4 file

    output_directory = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )
    output_filename = f"{country}_ASOS_filter_{year}.nc"
    output_path = os.path.join(output_directory, output_filename)
    ds_adjusted.to_netcdf(output_path)
    print(f"{output_filename} done!")


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
merge_csv_station(country, data_folder, data_save_category, output_folder)
year_list = get_year(start_year=1979, end_year=2023)
csv_paths = get_asos_year_file_list(
    country, data_folder, data_save_category, output_folder
)
for year, csv_path in tqdm(zip(year_list, csv_paths)):
    csv_to_nc4(csv_path, year, country, data_folder, data_save_category, output_folder)
