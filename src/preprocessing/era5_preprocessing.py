# Name: Wenqi Wang
# Github username: acse-ww721

import os
import re
import xarray as xr
import numpy as np
import xesmf as xe
from utils import folder_utils
from tqdm import tqdm

"""
Running on the Windows 11 system
because xesmf is not supported on M1 Silicon Mac

lat: 57.75 - 50.00 (32)
lon: -6 + 1.875 (32)

ddeg_out_lat= 0.25 (32)
ddeg_out_lon = 0.125 (64)
"""

PREFIX = "era5_pressure_level_"
SUFFIX = "_850.nc"


def get_era5_list(country, data_folder, data_category, output_folder):
    """
    Get a list of ERA5 NetCDF files for a specific country.

    This function searches for NetCDF files with "era5" in their names within the specified input folder path,
    and returns a list of the file paths for ERA5 data files associated with the specified country.

    Args:
        country (str): The country code or identifier.
        data_folder (str): The path to the data folder.
        data_category (str): The data category.
        output_folder (str): The output folder name.

    Returns:
        list: A list of file paths for ERA5 NetCDF data files.

    Example:
        >>> country = "GB"
        >>> data_folder = "data"
        >>> data_category = "processed_data"
        >>> output_folder = "ERA5_DATA"
        >>> era5_files = get_era5_list(country, data_folder, data_category, output_folder)
        # Retrieves a list of ERA5 NetCDF file paths for the specified country.

    """
    input_folder_path = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )
    nc_files = [
        f for f in os.listdir(input_folder_path) if f.endswith(".nc") and "era5" in f
    ]
    return [
        os.path.join(input_folder_path, nc_file) for nc_file in nc_files
    ]  # list for era5 nc files path


def merge_ds_by_year(era5_list, country, data_folder, data_category, output_folder):
    """
    Merge NetCDF files by year in the given list and save the merged files.

    This function organizes NetCDF files by year, merges files for each year, and saves the merged files with a specific prefix.
    It returns a list of unique years for which data has been merged.

    Args:
        era5_list (list): A list of file paths to ERA5 NetCDF data files.
        country (str): The country code or identifier.
        data_folder (str): The path to the data folder.
        data_category (str): The data category.
        output_folder (str): The output folder name.

    Returns:
        list: A list of unique years for which data has been merged.

    Example:
        >>> era5_list = ["file1.nc", "file2.nc"]
        >>> country = "US"
        >>> data_folder = "data"
        >>> data_category = "era5"
        >>> output_folder = "output"
        >>> merged_years = merge_ds_by_year(era5_list, country, data_folder, data_category, output_folder)
        # Merges ERA5 data files by year and saves the merged files.

    """

    # Organize files by year
    files_by_year = {}
    pattern = r"era5_pressure_level_(\d{4})_\d{2}_\d{3}\.nc"

    for file in era5_list:
        match = re.search(pattern, file)
        if match:
            year = match.group(1)
            if year not in files_by_year:
                files_by_year[year] = []
            files_by_year[year].append(file)

    # Merge and save for each year
    # Calculate the output folder outside the loop
    output_folder_path = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )

    for year, file_list in files_by_year.items():
        merged_ds = xr.open_mfdataset(file_list, combine="by_coords")
        output_filename = os.path.join(output_folder_path, f"{PREFIX}{year}{SUFFIX}")
        merged_ds.to_netcdf(output_filename)
        print(f"Merged data for {year} saved as {output_filename}")

    # Get unique years
    unique_years = list(files_by_year.keys())
    return unique_years


def cutoff_ds(era5_nc_path, lat_min, lat_max, lon_min, lon_max):
    """
    Cut off the dataset with the given latitude and longitude range.

    This function opens an ERA5 NetCDF dataset from the specified path, selects a subset of data within the specified latitude and longitude range, and returns the cut-off dataset.

    Args:
        era5_nc_path (str): The path to the ERA5 NetCDF dataset.
        lat_min (float): Minimum latitude for the cutoff.
        lat_max (float): Maximum latitude for the cutoff.
        lon_min (float): Minimum longitude for the cutoff.
        lon_max (float): Maximum longitude for the cutoff.

    Returns:
        xarray.Dataset: The cut-off dataset containing data within the specified latitude and longitude range.

    Example:
        >>> era5_nc_path = "era5_data.nc"
        >>> lat_min = 40.0
        >>> lat_max = 60.0
        >>> lon_min = -10.0
        >>> lon_max = 10.0
        >>> cut_ds = cutoff_ds(era5_nc_path, lat_min, lat_max, lon_min, lon_max)
        # Opens the ERA5 dataset, selects data within the specified latitude and longitude range, and returns the cut-off dataset.

    """
    with xr.open_dataset(era5_nc_path) as ds:
        return ds.sel(
            latitude=slice(
                lat_max, lat_min
            ),  # Reversed latitudes due to the era5 settings
            longitude=slice(lon_min, lon_max),
        )


# The regrid() function implementation below is a modification based on WeatherBench's GitHub code
# Original code link: https://github.com/pangeo-data/WeatherBench/blob/master/src/regrid.py
def regrid(ds_in, ddeg_out_lat, ddeg_out_lon, method="bilinear", reuse_weights=False):
    """
    Regrid a dataset horizontally (in the longitude direction).

    This function regrids an input xarray dataset to a new grid with the specified latitude and longitude resolutions using a specified regridding method.

    Args:
        ds_in (xarray.Dataset): The input xarray dataset.
        ddeg_out_lat (float): The output resolution for latitude.
        ddeg_out_lon (float): The output resolution for longitude.
        method (str, optional): The regridding method. Default is "bilinear".
        reuse_weights (bool, optional): Whether to reuse existing weights for regridding. Default is False.

    Returns:
        xarray.Dataset: The regridded dataset.

    Example:
        >>> ds_in = xr.open_dataset("input_data.nc")
        >>> ddeg_out_lat = 1.0
        >>> ddeg_out_lon = 1.0
        >>> method = "bilinear"
        >>> reuse_weights = False
        >>> ds_out = regrid(ds_in, ddeg_out_lat, ddeg_out_lon, method, reuse_weights)
        # Regrids the input dataset to the specified output resolution using the specified regridding method.

    """
    # Rename to ESMF compatible coordinates
    if "latitude" in ds_in.coords:
        ds_in = ds_in.rename({"latitude": "lat", "longitude": "lon"})

    # Create output grid
    grid_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(50, 58, ddeg_out_lat)),  # 50 - 57.75
            "lon": (["lon"], np.arange(-6, 2.0, ddeg_out_lon)),  # -6 - 1.875
        }
    )

    # Create regridder
    regridder = xe.Regridder(
        ds_in,
        grid_out,
        method,
        periodic=False,  # For a specific region, set periodic to False
        reuse_weights=False,  # recalculat weight automatically
    )

    # Hack to speed up regridding of large files
    ds_list = []
    chunk_size = 500
    n_chunks = len(ds_in.time) // chunk_size + 1
    for i in range(n_chunks):
        ds_small = ds_in.isel(time=slice(i * chunk_size, (i + 1) * chunk_size))
        ds_list.append(regridder(ds_small).astype("float32"))
    ds_out = xr.concat(ds_list, dim="time")

    # Set attributes since they get lost during regridding
    for var in ds_out:
        ds_out[var].attrs = ds_in[var].attrs
    ds_out.attrs.update(ds_in.attrs)

    # Regrid dataset
    # ds_out = regridder(ds_in)
    return ds_out


def save_regridded_era5(
    ds_out, year, country, data_folder, data_category, output_folder
):
    """
    Save regridded ERA5 data to a NetCDF file.

    This function saves a regridded ERA5 dataset to a NetCDF file with a specific naming convention.

    Args:
        ds_out (xarray.Dataset): The regridded ERA5 dataset.
        year (str): The year for which the data was regridded.
        country (str): The country code or identifier.
        data_folder (str): The path to the data folder.
        data_category (str): The data category.
        output_folder (str): The output folder name.

    Example:
        >>> ds_out = xr.open_dataset("regridded_era5_data.nc")
        >>> year = "2022"
        >>> country = "US"
        >>> data_folder = "data"
        >>> data_category = "era5"
        >>> output_folder = "output"
        >>> save_regridded_era5(ds_out, year, country, data_folder, data_category, output_folder)
        # Saves the regridded ERA5 data for the specified year to a NetCDF file.

    """
    prefix = "era5_pressure_level_"
    suffix = "_850.nc"
    output_folder = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )
    output_filename = os.path.join(output_folder, f"{prefix}{year}_regrid{suffix}")
    ds_out.to_netcdf(output_filename)
    print(f"Regridded data for {year} saved as {output_filename}")


def extract_T850_compute_mean_std(
    country, data_folder, data_category, output_folder, start_year=1979, end_year=2020
):
    """
    Extract T850 data, compute mean and standard deviation.

    This function extracts T850 data from regridded ERA5 NetCDF files for the specified year range, computes the mean and standard deviation of valid values, and returns the results.

    Args:
        country (str): The country code or identifier.
        data_folder (str): The path to the data folder.
        data_category (str): The data category.
        output_folder (str): The output folder name.
        start_year (int, optional): The start year for data extraction. Default is 1979.
        end_year (int, optional): The end year for data extraction. Default is 2020.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the T850 data.

    Example:
        >>> country = "US"
        >>> data_folder = "data"
        >>> data_category = "era5"
        >>> output_folder = "output"
        >>> start_year = 1979
        >>> end_year = 2020
        >>> mean_t850, std_t850 = extract_T850_compute_mean_std(country, data_folder, data_category, output_folder, start_year, end_year)
        # Extracts T850 data from specified years, computes mean and standard deviation.

    """
    # era5_pressure_level_2022_regrid_850.nc
    input_folder_path = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )
    nc_files = [
        os.path.join(input_folder_path, f)
        for f in os.listdir(input_folder_path)
        if f.endswith(".nc")
        and "regrid_850" in f
        and start_year <= int(f.split("_")[3]) <= end_year
    ]
    ds = xr.open_mfdataset(nc_files, combine="by_coords")

    t2m_data = ds["t"]

    mean_list = []
    std_list = []

    for chunk in tqdm(t2m_data):
        chunk_flatten = chunk.values.flatten()

        # Exclude NaN and zero values from calculation
        valid_values = chunk_flatten[~np.isnan(chunk_flatten) & (chunk_flatten != 0)]

        if len(valid_values) > 0:
            mean_list.append(np.nanmean(valid_values))
            std_list.append(np.nanstd(valid_values))

    mean_t2m = np.mean(mean_list)
    std_t2m = np.mean(std_list)

    return mean_t2m, std_t2m


def fill_nan_new(Z):
    """
    Fill NaN values in a 3D array using interpolation.

    This function fills NaN values in a 3D NumPy array using interpolation. It processes the start, end, and middle NaN values in each 2D slice of the array along the time dimension.

    Args:
        Z (numpy.ndarray): The 3D array containing NaN values to be filled.

    Returns:
        numpy.ndarray: The input array with NaN values filled using interpolation.

    Example:
        >>> Z = np.array([[[1.0, 2.0, np.nan, 4.0], [1.0, np.nan, 3.0, 4.0]], [[2.0, np.nan, 4.0, 5.0], [2.0, 3.0, 4.0, np.nan]]])
        >>> filled_Z = fill_nan_new(Z)
        # Fills NaN values in the input 3D array using interpolation.
    """
    for t in tqdm(range(Z.shape[0])):
        for i in range(Z.shape[1]):
            # Process the start nan of the array
            start = 0
            while start < Z.shape[2] and np.isnan(Z[t, i, start]):
                start += 1
            for j in range(start):
                Z[t, i, j] = Z[t, i, start] if start < Z.shape[2] else np.nan

            # Process the end nan of the array
            end = Z.shape[2] - 1
            while end >= 0 and np.isnan(Z[t, i, end]):
                end -= 1
            for j in range(end + 1, Z.shape[2]):
                Z[t, i, j] = Z[t, i, end] if end >= 0 else np.nan

            # Process the middle nan of the array
            for j in range(1, Z.shape[2] - 1):  # avoid the start and end
                if np.isnan(Z[t, i, j]):
                    prev_val = Z[t, i, j - 1]
                    next_val = Z[t, i, j + 1]

                    # only fill if both values are not nan
                    if not np.isnan(prev_val) and not np.isnan(next_val):
                        Z[t, i, j] = (prev_val + next_val) / 2
                    # if previous value is not nan, use previous value to fill
                    elif not np.isnan(prev_val):
                        Z[t, i, j] = prev_val
                    # if next value is not nan, use next value to fill
                    elif not np.isnan(next_val):
                        Z[t, i, j] = next_val
            Z[t, i, -1] = Z[t, i, -2]
    return Z


# Example usage

country = "GB"
data_folder = "data"
data_test_category = "test_data"
data_read_category = "raw_data"
data_save_category = "processed_data"
output_folder = "ERA5_DATA"
ddeg_out_lat = 0.25
ddeg_out_lon = 0.125

##################################################################################

# era5_list = []
era5_list = get_era5_list(
    country, data_folder, data_read_category, output_folder
)  # len = 528
# year_list = []
year_list = merge_ds_by_year(
    era5_list, country, data_folder, data_save_category, output_folder
)  # len = 44 (1979-2022)
merge_era5_list = get_era5_list(
    country, data_folder, data_save_category, output_folder
)  # len = 44 (1979-2022)

for merged_ds_path, year in tqdm(zip(merge_era5_list[0], year_list[0])):
    # ds = xr.open_dataset(merged_ds_path)
    ds = cutoff_ds(merged_ds_path, 50, 58, -6, 2)
    ds_out = regrid(ds, ddeg_out_lat, ddeg_out_lon)
    ds_out = ds_out.where((ds_out != 0) & ~np.isnan(ds_out), drop=True)
    ds_array = np.asarray(ds_out["t"])
    filled_array = fill_nan_new(ds_array)
    ds_out["t"] = xr.DataArray(
        filled_array, dims=ds_out["t"].dims, coords=ds_out["t"].coords
    )

    save_regridded_era5(
        ds_out, year, country, data_folder, data_save_category, output_folder
    )
    ds.close()
##################################################################################


mean_t2m, std_t2m = extract_T850_compute_mean_std(
    country,
    data_folder,
    data_save_category,
    output_folder,
    start_year=1979,
    end_year=2020,
)
