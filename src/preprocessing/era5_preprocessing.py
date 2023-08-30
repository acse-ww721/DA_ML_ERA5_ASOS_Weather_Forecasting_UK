import os
import re
import xarray as xr
import numpy as np
import xesmf as xe
import dask
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
def regrid(ds_in, ddeg_out_lat, ddeg_out_lon, method="bilinear", reuse_weights=False):
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


def extract_T850_compute_mean_std(
    country, data_folder, data_category, output_folder, start_year=1979, end_year=2020
):
    # era5_pressure_level_2022_regrid_850.nc
    input_folder_path = folder_utils.find_folder(country, data_folder, data_category, output_folder)
    nc_files = [
        os.path.join(input_folder_path, f)
        for f in os.listdir(input_folder_path)
        if f.endswith(".nc") and "regrid_850" in f and start_year <= int(f.split('_')[3]) <= end_year
    ]
    ds = xr.open_mfdataset(nc_files, combine="by_coords")

    t2m_data = ds['t']

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
    ds_array = np.asarray(ds_out['t'])
    filled_array=fill_nan_new(ds_array)
    ds_out['t'] = xr.DataArray(filled_array, dims=ds_out['t'].dims, coords=ds_out['t'].coords)

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
