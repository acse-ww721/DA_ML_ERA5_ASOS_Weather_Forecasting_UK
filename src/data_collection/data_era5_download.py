import cdsapi
import threading
import time
import os
from tqdm import tqdm
from utils import folder_utils, time_utils
from concurrent.futures import ThreadPoolExecutor  # thread pool module
from data_era5_t850 import data_year, data_month, data_day, data_time, area_uk

"""
Download ERA5 data from CDS
variables: t2m from 1979 to 2022
data level: hourly
data volume: 365*24*44 = 383040
"""


# folder setting
country = [
    "GB",
]
data_folder = "data"
data_category = "raw_data"
output_folder = "ERA5_DATA"


# variable setting
dataset = "reanalysis-era5-single-levels"

variable_list = [
    "2m_temperature",
]

# c = cdsapi.Client()


def era5_get_data_single_level(c, dataset, variable_list, year):
    # multi threads version may have figure in overcapacity error
    # c: api_server
    # dataset: target dataset
    # variable_list: the target variable
    try:
        output_directory = folder_utils.create_folder(
            country, data_folder, data_category, output_folder  # i is the data_year
        )
        output_filename = f"era5_single_level_{year}.nc"
        output_filepath = os.path.join(output_directory, output_filename)
        c.retrieve(
            dataset,
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": variable_list,
                "year": year,
                "month": data_month,
                "day": data_day,
                "time": data_time,
                # 'format': 'netcdf.zip',
                "area": area_uk,  # the UK range
            },
            output_filepath,
        )

        print(f"{output_filename} done!")

    except Exception as e:
        print(f"Error downloading {output_filename}: {e}\n")


# Multiple threads module for accelerating

"""
Download t2m data from ASOS
Variables: T850, RH850. T1000, RH1000
Time range: 1979-2022
data level: hourly
data volume: 1,571,328 
"""


def thread_function(year):
    c = cdsapi.Client()  # Initialize client within the thread

    start_time = time.time()  # Record start time
    era5_get_data_single_level(
        c,
        dataset,
        variable_list,
        year,
    )
    end_time = time.time()  # Record end time
    run_time = end_time - start_time
    print(f"Download time: {run_time:.3f} s")


# Create a thread pool  # 8 threads
with ThreadPoolExecutor(max_workers=8) as executor:
    # iterate through the data_year and pressure_level
    for i in tqdm(data_year):
        executor.submit(thread_function, i)


# Single thread module for solving long time queue problem
def era5_get_data_single_level_st(c, dataset, variable_list, year, month):
    # single thread version
    # c: api_server
    # dataset: target dataset
    # variable_list: the target variable
    try:
        output_directory = folder_utils.create_folder(
            country, data_folder, data_category, output_folder  # i is the data_year
        )
        output_filename = f"era5_single_level_{year}.nc"
        output_filepath = os.path.join(output_directory, output_filename)
        c.retrieve(
            dataset,
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": "2m_temperature",
                "year": year,
                "month": month,
                "day": time_utils.days_check(year, month),
                "time": data_time,
                # 'format': 'netcdf.zip',
                "area": area_uk,  # the UK range
            },
            output_filepath,
        )

        print(f"{output_filename} done!")

    except Exception as e:
        print(f"Error downloading {output_filename}: {e}\n")


def thread_function_st(year, month):
    c = cdsapi.Client()  # Initialize client within the thread

    start_time = time.time()  # Record start time
    era5_get_data_single_level(
        c,
        dataset,
        variable_list,
        year,
        month,
    )
    end_time = time.time()  # Record end time
    run_time = end_time - start_time
    print(f"Download time: {run_time:.3f} s")


# Example usage

for i in data_year:
    for k in data_month:
        thread_function(i, k)
