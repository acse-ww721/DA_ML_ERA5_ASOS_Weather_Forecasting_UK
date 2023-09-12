# Name: Wenqi Wang
# Github username: acse-ww721

# The European Union has 27 member states
# The site use GB to denote UK_ASOS
# the eu_member_codes has 28 variables: UK + EU

import os
import time
import datetime
import requests
import pandas as pd
import concurrent.futures
from bs4 import BeautifulSoup

# from tqdm import tqdm
from utils import folder_utils

"""
Download t2m data from ASOS
Variables: t2m
Time range: 1979-2022
data level: hourly
data volume: 365*24*44 = 383040 
"""


eu_member_codes = [
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
    "GB",
]

# Extract data time range
startts = datetime.datetime(2022, 1, 1)
endts = datetime.datetime(2023, 8, 2)

# Number of attempts to download data
MAX_ATTEMPTS = 6


def get_all_network():
    """
    Get a list of option values from a select element on a web page.

    This function sends a GET request to the specified URL, parses the HTML content,
    and extracts the values of the "option" elements within the first "select" element
    found on the page.

    Example:
        >>> get_all_network()
        # Prints the option values from the specified web page.

    Note:
        You need to have the 'requests' and 'beautifulsoup4' libraries installed
        to use this function.

    """
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
    """
    Get valid network URLs for a list of countries.

    This function generates network URLs for a list of countries and checks if the URLs are valid by sending a HEAD request.
    Valid URLs are added to the 'valid_urls' list, while invalid URLs are added to the 'invalid_urls' list.

    Args:
        country_list (list): A list of country codes or identifiers.

    Returns:
        list: A list of valid network URLs.

    Example:
        >>> country_list = ["GB", "FR", "DE"]
        >>> valid_urls = get_network_url(country_list)
        # Returns a list of valid network URLs for the specified countries.

    """
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


def get_all_station_by_network(country_list, data_folder, data_category, output_folder):
    """
    Get station information for a list of countries by network and save it as CSV files.

    This function retrieves station information in GeoJSON format for each country in the list,
    extracts relevant data, and saves it as a CSV file in the specified output directory.

    Args:
        country_list (list): A list of country codes or identifiers.
        data_folder (str): The path to the data folder.
        data_category (str): The data category.
        output_folder (str): The output folder name.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing station information for the last country in the list.

    Example:
        >>> country_list = ["GB", "FR", "DE"]
        >>> data_folder = "data"
        >>> data_category = "raw_data"
        >>> output_folder = "ASOS_DATA"
        >>> station_data_df = get_all_station_by_network(country_list, data_folder, data_category, output_folder)
        # Retrieves station information for the specified countries and saves CSV files.

    """
    # valid_urls = get_network_url()
    for i in country_list:
        url_station_geojson = (
            f"https://mesonet.agron.iastate.edu/geojson/network/{i}__ASOS.geojson"
        )
        output_directory = folder_utils.create_folder(
            i, data_folder, data_category, output_folder
        )
        output_filename = f"{i}__asos_station_network.csv"
        output_filepath = os.path.join(output_directory, output_filename)

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
        df.to_csv(output_filepath, index=False, encoding="utf-8")

    return df


def get_data_url(df, startts, endts):
    """
    Generate data download URLs for a DataFrame of stations and date range.

    This function generates data download URLs for each station in the DataFrame, for a specified date range.
    The URLs are constructed based on the station ID, start date, and end date.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing station information, including the "ID" column.
        startts (datetime.pyi): The start date for the data download.
        endts (datetime.pyi): The end date for the data download.

    Returns:
        list: A list of data download URLs corresponding to each station ID.
        list: A list of station IDs.

    Example:
        >>> import pandas as pd
        >>> station_data = pd.read_csv("stations.csv")
        >>> startts = pd.Timestamp("2023-01-01")
        >>> endts = pd.Timestamp("2023-12-31")
        >>> url_list, id_list = get_data_url(station_data, startts, endts)
        # Generates data download URLs for the specified date range and station IDs.

    """
    url_site_list = []
    id_list = []
    url_site_header = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    url_site_tail = f"data=tmpf&"  # add all data variables / f"data=tmpc&"
    url_site_tail += startts.strftime("year1=%Y&month1=%m&day1=%d&")  # add start date
    url_site_tail += endts.strftime("year2=%Y&month2=%m&day2=%d&")  # add end date
    url_site_tail += f"tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=null&trace=T&direct=no&report_type=3&report_type=4"  # add data format
    for id in df["ID"]:
        url_site = f"{url_site_header}station={id}&{url_site_tail}"  # add all stations
        url_site_list.append(url_site)
        id_list.append(id)

    return url_site_list, id_list  # return station list


def download_data(url_site):
    """
    Download data from a given URL with retry attempts.

    This function sends GET requests to the specified URL and retries up to a maximum number of attempts (MAX_ATTEMPTS)
    in case of connection errors or if the response starts with "ERROR."

    Args:
        url_site (str): The URL to download data from.

    Returns:
        str: The downloaded data as a string, or an empty string if download attempts are exhausted.

    Example:
        >>> url = "https://example.com/data"
        >>> data = download_data(url)
        # Downloads data from the specified URL with retry attempts.

    """
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


def save_data(
    url_site,
    country,
    station,
    startts,
    endts,
    data_folder,
    data_category,
    output_folder,
):
    """
    Download and save data from a URL as a CSV file.

    This function downloads data from the specified URL, parses it, and saves it as a CSV file in the specified output directory.
    The CSV file is named based on the country, station, and date range.

    Args:
        url_site (str): The URL to download data from.
        country (str): The country code or identifier.
        station (str): The station identifier.
        startts (datetime.pyi): The start date of the data.
        endts (datetime.pyi): The end date of the data.
        data_folder (str): The path to the data folder.
        data_category (str): The data category.
        output_folder (str): The output folder name.

    Example:
        >>> url = "https://example.com/data"
        >>> country = "US"
        >>> station = "ABC123"
        >>> start_date = pd.Timestamp("2023-01-01")
        >>> end_date = pd.Timestamp("2023-12-31")
        >>> data_folder = "data"
        >>> data_category = "temperature"
        >>> output_folder = "output"
        >>> save_data(url, country, station, start_date, end_date, data_folder, data_category, output_folder)
        # Downloads and saves data from the specified URL as a CSV file.

    """
    data = download_data(url_site)
    # output_filename = f"{country}_{startts:%Y%m%d%H%M}_{endts:%Y%m%d%H%M}.csv"
    output_directory = folder_utils.create_folder(
        country, data_folder, data_category, output_folder
    )
    output_filename = f"{country}_{station}_{startts:%Y%m%d}_{endts:%Y%m%d}.csv"
    output_filepath = os.path.join(output_directory, output_filename)
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
    df.to_csv(output_filepath, index=False, encoding="utf-8")
    print(f"{output_filename} done!")


def download_and_save_data(
    url_site,
    country,
    station,
    startts,
    endts,
    data_folder,
    data_category,
    output_folder,
):
    """
    Download data from a URL, save it as a CSV file, and measure download time.

    This function combines the download and save operations, measures the download time, and prints download status.

    Args:
        url_site (str): The URL to download data from.
        country (str): The country code or identifier.
        station (str): The station identifier.
        startts (datetime.pyi): The start date of the data.
        endts (datetime.pyi): The end date of the data.
        data_folder (str): The path to the data folder.
        data_category (str): The data category.
        output_folder (str): The output folder name.

    Example:
        >>> url = "https://example.com/data"
        >>> country = "GB"
        >>> station = "EGAA"
        >>> startts = datetime.datetime(1976, 1, 1)
        >>> endts = end_date = datetime.datetime(2023, 1, 1)
        >>> data_folder = "data"
        >>> data_category = "raw_data"
        >>> output_folder = "ASOS_DATA"

        >>> download_and_save_data(url, country, station, start_date, end_date, data_folder, data_category, output_folder)
        # Downloads data from the specified URL, saves it as a CSV file, and measures download time.

    """
    start_time = time.time()  # Record start time
    data = download_data(url_site)
    end_time = time.time()  # Record end time
    download_time = end_time - start_time

    if data:
        save_data(
            url_site,
            country,
            station,
            startts,
            endts,
            data_folder,
            data_category,
            output_folder,
        )
        print(f"{station} - Download time: {download_time:.3f} s")
    else:
        print(f"{station} - Download failed")


def download_and_save_data_thread(args):
    url_site, country, station_id, startts, endts = args
    download_and_save_data(
        url_site,
        country,
        station_id,
        startts,
        endts,
        data_folder,
        data_category,
        output_folder,
    )


# UK example
country = [
    "GB",
]

data_folder = "data"
data_category = "raw_data"
output_folder = "ASOS_DATA"

start_date = datetime.datetime(1976, 1, 1)
end_date = datetime.datetime(
    2023, 1, 1
)  # the end date is 2022/12/31 because the end date is not included

gb_df = get_all_station_by_network(country, data_folder, data_category, output_folder)
url_site_list, id_list = get_data_url(gb_df, start_date, end_date)
args_list = [
    (url_site, "GB", station_id, start_date, end_date)
    for url_site, station_id in zip(url_site_list, id_list)
]

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(download_and_save_data_thread, args_list)  # fast
    time.sleep(10)  # wait for 10 seconds to avoid the server overcapacity error


# test url 1:
"""
https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?
station=EGAA&data=all&year1=2023&month1=8&day1=14&year2=2023&month2=8&day2=15&tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=M&trace=T
&direct=no&report_type=3&report_type=4
"""
# test url 2:

"""
https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?
station=EGAC&data=tmpf&year1=1976&month1=1&day1=1&year2=2022&month2=12&day2=31&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=null&trace=null
&direct=no&report_type=3&report_type=4
"""
