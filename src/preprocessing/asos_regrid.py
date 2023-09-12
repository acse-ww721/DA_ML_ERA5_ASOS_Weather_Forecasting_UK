# Name: Wenqi Wang
# Github username: acse-ww721

import os
import pandas as pd
import numpy as np
import gstools as gs
from utils import folder_utils
from tqdm import tqdm
from asos_preprocessing import (
    get_year,
    get_asos_year_file_list,
    filter_data,
)


def krige_regrid_poly(
    year_df_path, year, country, data_folder, data_category, output_folder
):
    """
    Perform kriging interpolation with polynomial drift modeling to regrid meteorological data.

    Args:
        year_df_path (str): Path to the input CSV file for the specified year.
        year (int): The year for which the data is being regridded.
        country (str): Country code or name.
        data_folder (str): Folder where the data is stored.
        data_category (str): Category of the data.
        output_folder (str): Folder where the regridded data will be saved.

    Returns:
        None

    Example:
        >>> krige_regrid_poly("year_data.csv", 2022, "GB", "data_folder", "data_category", "output_folder")
        # Performs kriging interpolation and polynomial drift modeling for the specified year.
    """
    # 1. Load the data
    df = pd.read_csv(year_df_path)
    df = filter_data(df)
    # lat = df["latitude"].values
    # lon = df["longitude"].values

    # 2. Create a new dataframe to store the interpolated data
    output_df = pd.DataFrame()

    # 3.Define the grid
    g_lon = np.linspace(-6.0, 1.875, 64)  # longitude
    g_lat = np.linspace(50.0, 57.75, 32)  # latitude
    # gridx, gridy = np.meshgrid(gridx, gridy)

    # # 4. Drift term
    # def north_south_drift(lat, lon):
    #     return lat
    #
    #     # 4. Drift term
    #
    # def polynomial_drift(lat, lon):
    #     return [1, lat, lon, lat**2, lon**2, lat * lon]

    # 4. Drift terms

    def drift_1(lat, lon):
        return 1

    def drift_lat(lat, lon):
        return lat

    def drift_lon(lat, lon):
        return lon

    def drift_lat2(lat, lon):
        return lat**2

    def drift_lon2(lat, lon):
        return lon**2

    def drift_lat_lon(lat, lon):
        return lat * lon

    drift_functions = [
        drift_1,
        drift_lat,
        drift_lon,
        drift_lat2,
        drift_lon2,
        drift_lat_lon,
    ]

    unique_times = df["time"].unique()

    # Iterate over each time
    for time_point in tqdm(unique_times):
        # 1. Load data for the specific time point
        time_df = df[df["time"] == time_point]
        t2m = time_df["t2m"].values
        lat = time_df["latitude"].values
        lon = time_df["longitude"].values

        # 2. Estimate the variogram
        bin_center, vario = gs.vario_estimate(
            (lat, lon), t2m, latlon=True, geo_scale=gs.KM_SCALE, max_dist=900
        )

        # 3. krige interpolation
        model = gs.Spherical(latlon=True, geo_scale=gs.KM_SCALE)
        model.fit_variogram(bin_center, vario, nugget=False)

        # 5. Universal Kriging

        uk = gs.krige.Universal(
            model=model,
            cond_pos=(lat, lon),
            cond_val=t2m,
            drift_functions=drift_functions,
        )

        uk.set_pos((g_lat, g_lon), mesh_type="structured")
        interpolated_values = uk(return_var=False)

        temp_df = pd.DataFrame(
            {
                "lat": np.tile(g_lat, len(g_lon)),
                "lon": np.repeat(g_lon, len(g_lat)),
                "time": [time_point] * len(g_lat) * len(g_lon),
                "t2m": interpolated_values.ravel(),
            }
        )

        output_df = pd.concat([output_df, temp_df], ignore_index=True)

    # Save to csv file
    output_directory = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )
    output_filename = f"{country}_ASOS_krige_{year}.csv"
    output_path = os.path.join(output_directory, output_filename)
    output_df.to_csv(output_path, index=False)

    # return output_df


########################################################################################

# Example usage
country = "GB"
data_folder = "data"
data_read_category = "raw_data"
data_test_category = "test_data"
data_save_category = "processed_data"
output_folder = "ASOS_DATA"

year_list = get_year(start_year=1979, end_year=2023)
csv_paths = get_asos_year_file_list(
    country, data_folder, data_save_category, output_folder
)
for year, csv_path in tqdm(zip(year_list, csv_paths)):
    krige_regrid_poly(
        csv_path, year, country, data_folder, data_save_category, output_folder
    )
