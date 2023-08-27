import os
import re
import xarray as xr
import numpy as np
import xesmf as xe
import dask
from utils import folder_utils
from tqdm import tqdm

"""V4 
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
    Merge netcdf files by year in the given list.
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
    Cut off the dataset with given lat and lon range
    :param era5_nc_path: Input xarray dataset path
    :param lat_min: Minimum latitude
    :param lat_max: Maximum latitude
    :param lon_min: Minimum longitude
    :param lon_max: Maximum longitude
    :return: ds: Cut off dataset
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
def regrid(ds_in, ddeg_out_lat, ddeg_out_lon, method="bilinear", reuse_weights=True):
    """
    Regrid horizontally (longitude direction).
    :param ds_in: Input xarray dataset
    :param ddeg_out_lat: Output resolution latitude
    :param ddeg_out_lon: Output resolution longitude
    :param method: Regridding method
    :param reuse_weights: Reuse weights for regridding
    :return: ds_out: Regridded dataset
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
    prefix = "era5_pressure_level_"
    suffix = "_850.nc"
    output_folder = folder_utils.find_folder(
        country, data_folder, data_category, output_folder
    )
    output_filename = os.path.join(output_folder, f"{prefix}{year}_regrid{suffix}")
    ds_out.to_netcdf(output_filename)
    print(f"Regridded data for {year} saved as {output_filename}")


# Example usage

country = "GB"
data_folder = "data"
data_test_category = "test_data"
data_read_category = "raw_data"
data_save_category = "processed_data"
output_folder = "ERA5_DATA"
ddeg_out_lat = 0.25
ddeg_out_lon = 0.125

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


for merged_ds_path, year in tqdm(zip(merge_era5_list, year_list)):
    ds = xr.open_dataset(merged_ds_path)
    ds = cutoff_ds(merged_ds_path, 50, 58, -6, 2)
    ds_out = regrid(ds, ddeg_out_lat, ddeg_out_lon)
    save_regridded_era5(
        ds_out, year, country, data_folder, data_save_category, output_folder
    )
    ds.close()
