import cdsapi
import threading
import time
from tqdm import tqdm
from data_era5_download import days_check
from data_era5_download import data_year, data_month, data_time

# 1,149.904 > 120,000

dataset = 'reanalysis-era5-pressure-levels'
variable_list = [
    'geopotential',
    'relative_humidity',
    'specific_humidity',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'vertical_velocity',
]

pressure_level = [
    '50',
    '100',
    '150',
    '200',
    '300',
    '400',
    '500',
    '600',
    '700',
    '850',
    '925',
    '1000',
]

download_times = {}  # Dictionary to store download times for each task

c = cdsapi.Client()


def era5_get_data_keisler(c, dataset, variable_list, year, month, pressure_level):
    try:
        start_time = time.time()  # Record start time
        filename = f'era5_pressure_level_{year}_{month}_{pressure_level}.nc'
        c.retrieve(
            dataset,
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variable_list,
                'pressure_level': pressure_level,
                'year': year,
                'month': month,
                'day': days_check(year, month),
                'time': data_time,
                'area': [
                    61,
                    -8,
                    50,
                    2,
                ],
            },
            filename,
        )
        end_time = time.time()  # Record end time
        download_time = end_time - start_time
        download_times[(year, month)] = download_time

        print(f'{filename} done!')
        print(f'Download time: {download_time:.3f} s')

    except Exception as e:
        print(f'Error downloading {filename}: {e}\n')


# Multiple threads module for accelerating


def thread_function(year, month, pressure_level):
    era5_get_data_keisler(
        c,
        dataset,
        variable_list,
        year,
        month,
        pressure_level,
    )


threads = []

for i in data_year:
    for j in tqdm(data_month):
        for k in tqdm(pressure_level):
            thread = threading.Thread(target=thread_function, args=(i, j, k))
            threads.append(thread)
            thread.start()

for thread in threads:
    thread.join()

# Calculate and print total download times
total_download_time = sum(download_times.values())
print(f'Total download time: {total_download_time:.2f} seconds\n')
