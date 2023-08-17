import os
import pandas as pd
from utils import folder_utils
from tqdm import tqdm


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
        if f.endswith('.csv') and "asos_station_network" not in f
    ]
    # Read and merge the csv files in queue
    for csv_file in tqdm(csv_files):
        csv_file_path = os.path.join(input_folder_path, csv_file)
        df = pd.read_csv(csv_file_path)
        raw_df = pd.concat([raw_df, df], ignore_index=True)

    return raw_df


def process_asos_rawdata(df):
    # Split "valid" column into "date" and "time" columns
    df['date'] = (
        df['valid'].str.split(' ', expand=True)[0].str.replace('-', '').astype(int)
    )
    df['time'] = df['valid'].str.split(' ', expand=True)[1]

    # Convert Fahrenheit to Celsius for "tmpf", "feel" and "dwpf" columns
    df['tmpf'] = (df['tmpf'] - 32) * 5 / 9
    df['tmpf'] = df['tmpf'].round(1)

    df['dwpf'] = (df['dwpf'] - 32) * 5 / 9
    df['dwpf'] = df['dwpf'].round(1)

    # Convert Fahrenheit to Celsius for "feel" column
    df['feel'] = (df['feel'] - 32) * 5 / 9
    df['feel'] = df['feel'].round(1)

    # Convert knots to m/s for "sknt" and "gust" columns
    df['sknt'] = df['sknt'] * 0.514444
    df['gust'] = df['gust'] * 0.514444

    # Convert inches to meters for "p01i" and "alti" columns
    df['p01i'] = df['p01i'] * 0.0254
    df['alti'] = df['alti'] * 0.0254

    # Convert millibar to Pa for "mslp" column
    df['mslp'] = df['mslp'] * 100

    # Drop columns
    columns_to_drop = [
        'valid',
        'skyc1',
        'skyc2',
        'skyc3',
        'skyc4',
        'skyl1',
        'skyl2',
        'skyl3',
        'skyl4',
        'wxcodes',
        'ice_accretion_1hr',
        'ice_accretion_3hr',
        'ice_accretion_6hr',
        'peak_wind_gust',
        'peak_wind_drct',
        'peak_wind_time',
        'metar',
        'snowdepth',
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    return df


def save_asos_processed_data(
    processed_df, country, data_folder, data_category, output_folder
):
    output_directory = folder_utils.create_folder(
        country, data_folder, data_category, output_folder
    )
    output_filename = f"{country}_ASOS_processed_data.csv"
    output_filepath = os.path.join(output_directory, output_filename)
    processed_df.to_csv(output_filepath, index=False, encoding="utf-8")
    print(f'{output_filename} done!')


# Example usage

country = "GB"
data_folder = "data"
data_read_category = "raw_data"
data_save_category = "processed_data"
output_folder = "ASOS_DATA"

raw_df = extract_data_to_df(country, data_folder, data_read_category, output_folder)
processed_df = process_asos_rawdata(raw_df)
save_asos_processed_data(
    processed_df, country, data_folder, data_save_category, output_folder
)
