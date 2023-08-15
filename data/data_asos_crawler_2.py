# The European Union has 27 member states
# The site use GB to denote UK_ASOS
# the eu_member_codes has 28 variables: UK + EU

import requests
import pandas as pd
import datetime
import time
from bs4 import BeautifulSoup
from urllib.request import urlopen

eu_member_codes = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
    'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
    'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'GB',
]

# Extract data time range
startts =  datetime.datetime(2022,1,1)
endts =  datetime.datetime(2023,8, 2)

# Number of attempts to download data
MAX_ATTEMPTS = 6

def get_all_network():
    url = "https://mesonet.agron.iastate.edu/request/download.phtml?network=FR__ASOS"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    select_element = soup.find("select")
    if select_element:
        option_elements = select_element.find_all("option")

        option_values = [option["value"] for option in option_elements if option.get("value")]

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

def get_all_station_by_network(valid_urls):
    # valid_urls = get_network_url()
    for i in valid_urls:
        url_station_geojson = f"{i}.geojson"

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
            data.append(row)

        # Transfer the list to Pandas DataFrame
        df = pd.DataFrame(data)
    return df

def get_data_url(df,startts,endts):
    url_site ="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    for id in df['ID']:
        url_site+= f"station={id}&" # add all stations
        url_site+= f"data=all&" # add all data variables
        url_site+= startts.strftime("year1=%Y&month1=%m&day1=%d&") # add start date
        url_site+=  endts.strftime("year2=%Y&month2=%m&day2=%d&")  # add end date
        url_site+= f"tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=null&trace=T&direct=no&report_type=3&report_type=4"    # add data format

    return url_site

def download_data(url_data):
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            response = requests.get(url_data, timeout=300)
            data = response.text
            if data is not None and not data.startswith("ERROR"):
                return data
        except Exception as e:
            print(f"download_data({url_data}) failed with {e}")
            time.sleep(5)
        attempt += 1

    print("Exhausted attempts to download, returning empty data")
    return ""

def save_data(url_site,country,startts,endts):
    data = download_data(url_site)
    output_filename = f"{country}_{startts:%Y%m%d%H%M}_{endts:%Y%m%d%H%M}.csv"
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
    df.to_csv( output_filename , index=False, encoding="utf-8")

# UK example
country = ["GB",]
start_date =  datetime.datetime(2022,1,1)
endt_date =  datetime.datetime(2023,8, 2)

valid_url = get_network_url(country)
gb_df = get_all_station_by_network(valid_url)
url_site = get_data_url(gb_df,start_date,endt_date)
save_data(url_site,"GB",start_date,endt_date)







