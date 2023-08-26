import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from utils import folder_utils

"""V3"""


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


def merge_csv_station(country, data_folder, data_category, output_folder):
    # Find csvs
    input_folder = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )
    station_network_csv = "GB__asos_station_network.csv"
    asos_data_csv = "GB_ASOS_processed_data_lite.csv"
    station_network_csv_path = os.path.join(input_folder, station_network_csv)
    asos_data_csv_path = os.path.join(input_folder, asos_data_csv)

    # Read csvs
    station_id_df = pd.read_csv(station_network_csv_path)
    station_info_df = pd.read_csv(asos_data_csv_path)

    columns_to_drop = [
        "Country",
        "Network",
        "Archive_Begin",
        "Archive_End",
        "Time_Domain",
    ]

    station_id_df = station_id_df.drop(columns=columns_to_drop)
    station_id_df.rename(columns={"ID": "station"}, inplace=True)

    # Merge by "STATION"
    merged_df = pd.merge(station_id_df, station_info_df, on="station", how="left")

    return merged_df


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


# Example usage

country = "GB"
data_folder = "data"
data_read_category = "raw_data"
data_test_category = "test_data"
data_save_category = "processed_data"
output_folder = "ASOS_DATA"

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
