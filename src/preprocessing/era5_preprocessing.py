import os
import re
import xarray as xr
from utils import folder_utils
from tqdm import tqdm


def extract_year_month_from_filename(filename):
    match = re.search(r"(\d{4})_(\d{2})", filename)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        month = f"{month:02}"
        return year, month
    else:
        raise ValueError(f"Cannot extract year and month from filename: {filename}")


def save_era5_processed_data(
    merged_ds, year, month, country, data_folder, data_category, output_folder
):
    output_directory = folder_utils.create_folder(
        country, data_folder, data_category, output_folder
    )
    output_filename = f"{country}_ERA5_{year}_{month}_processed_data.nc"
    output_filepath = os.path.join(output_directory, output_filename)
    merged_ds.to_netcdf(output_filepath)
    print(f"{output_filename} done!")


# Now you have merged datasets for both single level and pressure level data
def extract_merge_nc_data(
    country, data_folder, data_read_category, data_save_category, output_folder
):
    input_folder_path = folder_utils.find_folder(
        country, data_folder, data_read_category, output_folder
    )

    # Get all the single_level data filename
    nc_files_single_level = [
        f
        for f in os.listdir(input_folder_path)
        if f.endswith(".nc") and "single_level" in f
    ]

    # Create a new blank dataset to store the merged data
    # merged_ds = xr.Dataset()

    # Loop single_level files
    for nc_file_A in tqdm(nc_files_single_level):
        nc_file_path_A = os.path.join(input_folder_path, nc_file_A)

        #
        ds_A = xr.open_dataset(nc_file_path_A)

        # extract the corresponding year and month
        year_A, month_A = extract_year_month_from_filename(nc_file_A)

        # Build the pressure level file path
        nc_file_B = f"era5_pressure_level_{year_A}_{month_A}_1000.nc"
        nc_file_path_B = os.path.join(input_folder_path, nc_file_B)

        #
        ds_B = xr.open_dataset(nc_file_path_B)

        # Merge single_level and pressure level by the key "time" and overwrite ds_A
        ds_A = ds_A.combine_first(ds_B)

        # Close all dataset
        ds_A.close()
        ds_B.close()

        save_era5_processed_data(
            ds_A,
            year_A,
            month_A,
            country,
            data_folder,
            data_save_category,
            output_folder,
        )


# Example usage

country = "GB"
data_folder = "data"
data_test_category = "test_data"
data_read_category = "raw_data"
data_save_category = "processed_data"
# output_folder = "ASOS_DATA"
output_folder2 = "ERA5_DATA"

#
extract_merge_nc_data(
    country, data_folder, data_read_category, data_save_category, output_folder2
)
