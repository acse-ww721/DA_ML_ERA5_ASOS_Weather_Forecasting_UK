import cdsapi
import threading
import time
from tqdm import tqdm

dataset1 = 'reanalysis-era5-land'
# dataset2 =''
data_year = [  # the target years
    '2022',
    '2023',
]
data_month = [  # the target months
    '01',
    '02',
    '03',
    '04',
    '05',
    '06',
    '07',
    '08',
    '09',
    '10',
    '11',
    '12',
]
data_time = [  # the target times_UTC
    '00:00',
    '01:00',
    '02:00',
    '03:00',
    '04:00',
    '05:00',
    '06:00',
    '07:00',
    '08:00',
    '09:00',
    '10:00',
    '11:00',
    '12:00',
    '13:00',
    '14:00',
    '15:00',
    '16:00',
    '17:00',
    '18:00',
    '19:00',
    '20:00',
    '21:00',
    '22:00',
    '23:00',
]
variable_list_1 = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_dewpoint_temperature',
    '2m_temperature',
    'skin_temperature',
]

download_times = {}  # Dictionary to store download times for each task

c = cdsapi.Client()


def is_leap_year(year):
    year = int(year)  # Convert the string to integer
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def days_check(year, month):
    year = int(year)  # Convert the string to integer
    days_by_month = {
        "01": 31,  # January
        "02": 29 if is_leap_year(year) else 28,  # February
        "03": 31,  # March
        "04": 30,  # April
        "05": 31,  # May
        "06": 30,  # June
        "07": 31,  # July
        "08": 31,  # August
        "09": 30,  # September
        "10": 31,  # October
        "11": 30,  # November
        "12": 31,  # December
    }

    return [str(day).zfill(2) for day in range(1, days_by_month.get(month, 0) + 1)]


def era5_get_data(c, dataset, variable_list, year, month):
    # c: api_server
    # dataset: target dataset
    # variable_list: the target variable
    try:
        start_time = time.time()  # Record start time
        filename = f'download_era5_land_{year}_{month}.netcdf.zip'
        c.retrieve(
            dataset,
            {
                'variable': variable_list,
                'year': year,
                'month': month,
                'day': days_check(year, month),
                'time': data_time,
                'format': 'netcdf.zip',
                'area': [
                    61,
                    -8,
                    50,
                    2,
                ],  # the UK range
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


# Mutiple threads module for accelerating


def thread_function(year, month):
    era5_get_data(c, dataset1, variable_list_1, year, month)


threads = []

for i in tqdm(data_year):
    for j in tqdm(data_month):
        thread = threading.Thread(target=thread_function, args=(i, j))
        threads.append(thread)
        thread.start()

for thread in threads:
    thread.join()

# Calculate and print total download times
total_download_time = sum(download_times.values())
print(f'Total download time: {total_download_time:.2f} seconds\n')
