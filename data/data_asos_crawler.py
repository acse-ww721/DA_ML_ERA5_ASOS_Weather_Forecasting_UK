# The European Union has 27 member states
# The site use GB to denote UK_ASOS
# the eu_member_codes has 28 variables: UK + EU

import time
import datetime
import requests
import pandas as pd
import concurrent.futures
from bs4 import BeautifulSoup
from tqdm import tqdm

eu_member_codes = [
    'AT',
    'BE',
    'BG',
    'HR',
    'CY',
    'CZ',
    'DK',
    'EE',
    'FI',
    'FR',
    'DE',
    'GR',
    'HU',
    'IE',
    'IT',
    'LV',
    'LT',
    'LU',
    'MT',
    'NL',
    'PL',
    'PT',
    'RO',
    'SK',
    'SI',
    'ES',
    'SE',
    'GB',
]

# Extract data time range
startts = datetime.datetime(2022, 1, 1)
endts = datetime.datetime(2023, 8, 2)

# Number of attempts to download data
MAX_ATTEMPTS = 6


def get_all_network():
    url = "https://mesonet.agron.iastate.edu/request/download.phtml?network=FR__ASOS"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    select_element = soup.find("select")
    if select_element:
        option_elements = select_element.find_all("option")

        option_values = [
            option["value"] for option in option_elements if option.get("value")
        ]

        print("Option values:")
        for value in option_values:
            print(value)
    else:
        print("Select element not found on the page.")


def get_network_url(country_list):
    valid_urls = []
    invalid_urls = []

    # for i in eu_member_codes:
    for i in country_list:
        url_network = f"https://mesonet.agron.iastate.edu/request/download.phtml?network={i}__ASOS"
        response = requests.head(url_network)

        if response.status_code == 200:
            valid_urls.append(url_network)
        else:
            invalid_urls.append(f"Invalid URL: {url_network}")

    return valid_urls


def get_all_station_by_network(country_list):
    # valid_urls = get_network_url()
    for i in country_list:
        url_station_geojson = (
            f"https://mesonet.agron.iastate.edu/geojson/network/{i}__ASOS.geojson"
        )

        # Get GeoJSON data
        response = requests.get(url_station_geojson)
        geojson_data = response.json()
        #  Create a list to store geojson
        data = []

        # Extract GeoJSON items
        for feature in geojson_data["features"]:
            properties = feature["properties"]
            geometry = feature["geometry"]
            row = {
                # "Type": feature.get("type", None),
                "Name": properties.get("sname", None),
                "ID": feature.get("id", None),
                "Latitude": geometry.get("coordinates", None)[1],
                "Logitude": geometry.get("coordinates", None)[0],
                "Elevation": properties.get("elevation", None),
                "Country": properties.get("country", None),
                "Network": properties.get("network", None),
                "Archive_Begin": properties.get("archive_begin", None),
                "Archive_End": properties.get("archive_end", None),
                "Time_Domain": properties.get("time_domain", None),
                # "Properties": properties
            }
            data.append(row)

        # Transfer the list to Pandas DataFrame
        df = pd.DataFrame(data)
    return df


def get_data_url(df, startts, endts):
    url_site_list = []
    id_list = []
    url_site_header = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    url_site_tail = f"data=all&"  # add all data variables
    url_site_tail += startts.strftime("year1=%Y&month1=%m&day1=%d&")  # add start date
    url_site_tail += endts.strftime("year2=%Y&month2=%m&day2=%d&")  # add end date
    url_site_tail += f"tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=null&trace=T&direct=no&report_type=3&report_type=4"  # add data format
    for id in df['ID']:
        url_site = f"{url_site_header}station={id}&{url_site_tail}"  # add all stations
        url_site_list.append(url_site)
        id_list.append(id)

    return url_site_list, id_list  # return station list


def download_data(url_site):
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            response = requests.get(url_site, timeout=300)
            data = response.text
            if data is not None and not data.startswith("ERROR"):
                return data
        except Exception as e:
            print(f"download_data({url_site}) failed with {e}")
            time.sleep(5)
        attempt += 1

    print("Exhausted attempts to download, returning empty data")
    return ""


def save_data(url_site, country, station, startts, endts):
    data = download_data(url_site)
    # output_filename = f"{country}_{startts:%Y%m%d%H%M}_{endts:%Y%m%d%H%M}.csv"
    output_filename = f"{country}_{station}_{startts:%Y%m%d}_{endts:%Y%m%d}.csv"
    # Split the data into lines
    lines = data.split("\n")

    # Extract column names from the first line
    column_names = lines[0].split(",")

    # Initialize lists to store data
    data_rows = []

    # Iterate through the remaining lines (data rows)
    for line in lines[1:]:
        if line:  # Skip empty lines
            data_row = line.split(",")
            data_rows.append(data_row)

    # Create a DataFrame from the data rows using the extracted column names
    df = pd.DataFrame(data_rows, columns=column_names)

    # Save the DataFrame to a CSV file
    df.to_csv(output_filename, index=False, encoding="utf-8")
    print(f'{output_filename} done!')

def download_and_save_data(url_site, country, station, startts, endts):
    start_time = time.time()  # Record start time
    data = download_data(url_site)
    end_time = time.time()  # Record end time
    download_time = end_time - start_time

    if data:
        save_data(data, country, station, startts, endts)
        print(f'{station} - Download time: {download_time:.3f} s')
    else:
        print(f'{station} - Download failed')


def download_and_save_data_thread(args):
    url_site, country, station_id, startts, endts = args
    download_and_save_data(url_site, country, station_id, startts, endts)

# UK example
country = [
    "GB",
]

start_date = datetime.datetime(2022, 1, 1)
end_date = datetime.datetime(2023, 8, 2)

gb_df = get_all_station_by_network(country)
url_site_list, id_list = get_data_url(gb_df, start_date, end_date)
for url_site, station_id in tqdm(zip(url_site_list, id_list)):
    download_and_save_data(url_site, "GB", station_id, start_date, end_date)


# test url:
"""
https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?
station=EGAA&data=all&year1=2023&month1=8&day1=14&year2=2023&month2=8&day2=15&tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=M&trace=T
&direct=no&report_type=3&report_type=4
"""
