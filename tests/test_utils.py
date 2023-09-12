# Name: Wenqi Wang
# Github username: acse-ww721

import pytest
import os
import numpy as np
import pandas as pd
from utils.folder_utils import get_current_directory, find_folder, create_folder
from utils.model_utils_tf import get_initial_weights
from utils.time_utils import time_select, is_leap_year, days_check


def test_get_current_directory():
    current_dir = get_current_directory()
    assert os.path.isabs(
        current_dir
    )  # Ensure that the returned path is an absolute path
    assert os.path.exists(current_dir)  # Ensure that the returned path exists


def test_find_folder():
    folder_path = find_folder("GB", "data", "raw_data", "ASOS_DATA")
    assert os.path.isabs(
        folder_path
    )  # Ensure that the returned path is an absolute path
    assert os.path.exists(folder_path)  # Ensure that the returned path exists


def test_create_folder(tmpdir):
    # Use a temporary directory for testing
    temp_dir = str(tmpdir)
    data_folder = os.path.join(temp_dir, "data")
    data_category = "raw_data"
    output_folder = "ASOS_DATA"
    c = "GB"

    folder_path = create_folder(c, data_folder, data_category, output_folder)

    assert os.path.isabs(
        folder_path
    )  # Ensure that the returned path is an absolute path
    assert os.path.exists(folder_path)  # Ensure that the created folder exists
    assert os.path.isdir(folder_path)  # Ensure that it is a directory


def test_get_initial_weights():
    output_size = 4
    weights = get_initial_weights(output_size)
    assert len(weights) == 2  # Ensure that the returned list contains two elements
    assert isinstance(
        weights[0], np.ndarray
    )  # Ensure that the first element is a numpy array
    assert isinstance(
        weights[1], np.ndarray
    )  # Ensure that the second element is a numpy array
    assert weights[0].shape == (
        output_size,
        6,
    )  # Ensure that the shape of the weight matrix is as expected
    assert weights[1].shape == (
        6,
    )  # Ensure that the shape of the bias vector is as expected
    assert np.all(
        weights[1][:2] == np.array([1.0, 0.0])
    )  # Ensure that the bias vector is initialized correctly


def test_time_select():
    data = {"Date": ["20220101", "20220102", "20220103"], "Value": [10, 15, 20]}
    df = pd.DataFrame(data)
    start_date = "20220102"
    end_date = "20220103"
    filtered_df = time_select(df, "Date", start_date, end_date)
    assert (
        len(filtered_df) == 2
    )  # Ensure that the filtered DataFrame has the expected number of rows
    assert (
        "Date" in filtered_df.columns
    )  # Ensure that the 'Date' column is present in the filtered DataFrame
    assert (
        "Value" in filtered_df.columns
    )  # Ensure that the 'Value' column is present in the filtered DataFrame
    assert (
        filtered_df["Date"] >= start_date
    ).all()  # Ensure that all dates are greater than or equal to the start_date
    assert (
        filtered_df["Date"] <= end_date
    ).all()  # Ensure that all dates are less than or equal to the end_date


def test_is_leap_year():
    assert is_leap_year(2020) is True  # 2020 is a leap year
    assert is_leap_year(2100) is False  # 2100 is not a leap year
    assert is_leap_year(2000) is True  # 2000 is a leap year (divisible by 400)
    assert (
        is_leap_year(1900) is False
    )  # 1900 is not a leap year (divisible by 4 and 100 but not by 400)


def test_days_check():
    assert days_check("2023", "02") == [
        str(day).zfill(2) for day in range(1, 29)
    ]  # February 2023 has 28 days (not a leap year)
    assert days_check("2020", "02") == [
        str(day).zfill(2) for day in range(1, 30)
    ]  # February 2020 has 29 days (leap year)
    assert days_check("2023", "04") == [
        str(day).zfill(2) for day in range(1, 31)
    ]  # April has 30 days
    assert days_check("2023", "13") == []  # Invalid month should return an empty list
