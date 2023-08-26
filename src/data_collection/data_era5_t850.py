import sys
import cdsapi
import threading
import time
import os

from tqdm import tqdm
from utils import time_utils, folder_utils
from concurrent.futures import ThreadPoolExecutor  # thread pool module


"""
Download ERA5 data from CDS
Variables: T850, RH850. T1000, RH1000
Time range: 1979-2022
data level: hourly
data volume: 1,571,328 
"""


# jupyter notebook setting
notebook_path = os.path.abspath("")
project_root = os.path.abspath(os.path.join(notebook_path, "../../"))
sys.path.append(project_root)

# folder setting
country = "GB"

data_folder = "data"
data_category = "raw_data"
output_folder = "ERA5_DATA"

# variable setting
data_year = [  # the target years
    "1979",
    "1980",
    "1981",
    "1982",
    "1983",
    "1984",
    "1985",
    "1986",
    "1987",
    "1988",
    "1989",
    "1990",
    "1991",
    "1992",
    "1993",
    "1994",
    "1995",
    "1996",
    "1997",
    "1998",
    "1999",
    "2000",
    "2001",
    "2002",
    "2003",
    "2004",
    "2005",
    "2006",
    "2007",
    "2008",
    "2009",
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
]

data_month = [  # the target months
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
]

data_day = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
]  # the target day  # s


data_time = [  # the target times_UTC
    "00:00",
    "01:00",
    "02:00",
    "03:00",
    "04:00",
    "05:00",
    "06:00",
    "07:00",
    "08:00",
    "09:00",
    "10:00",
    "11:00",
    "12:00",
    "13:00",
    "14:00",
    "15:00",
    "16:00",
    "17:00",
    "18:00",
    "19:00",
    "20:00",
    "21:00",
    "22:00",
    "23:00",
]

area_uk = [
    58,
    -7,
    50,
    2,
]

# 1,571,328 > 120,000 per time

# dataset setting
dataset = "reanalysis-era5-pressure-levels"
variable_list = [
    "temperature",
    "relative_humidity",
    # 'geopotential', the common variable for weather forecasting
    # "specific_humidity",
]

# pressure_level = [
#     "850",
#     "1000",
# ]

pressure_level = "850"

# create thread local storage object
thread_local = threading.local()


def era5_get_data_t850(c, dataset, variable_list, year, month):
    # download the whole year data
    try:
        output_directory = folder_utils.create_folder(
            country, data_folder, data_category, output_folder
        )
        output_filename = f"era5_pressure_level_{year}_{month}_{pressure_level}.nc"
        output_filepath = os.path.join(output_directory, output_filename)
        c.retrieve(
            dataset,
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": variable_list,
                "pressure_level": pressure_level,
                "year": year,
                "month": month,
                "day": time_utils.days_check(year, month),
                "time": data_time,
                "area": area_uk,
            },
            output_filepath,
        )

        print(f"{output_filename} done!")

    except Exception as e:
        print(f"Error downloading {output_filename}: {e}\n")


# Multiple threads module for accelerating


def thread_function(year, month):
    c = cdsapi.Client()  # Initialize client within the thread

    start_time = time.time()  # Record start time
    era5_get_data_t850(
        c,
        dataset,
        variable_list,
        year,
        month,
    )
    end_time = time.time()  # Record end time
    run_time = end_time - start_time
    print(f"Download time: {run_time:.3f} s")


# Create a thread pool  # 8 threads
with ThreadPoolExecutor(max_workers=2) as executor:
    # iterate through the data_year and data month to be the most efficient
    for i in tqdm(data_year):
        for k in tqdm(data_month):
            executor.submit(thread_function, i, k)
