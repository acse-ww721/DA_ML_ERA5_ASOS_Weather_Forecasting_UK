# Name: Wenqi Wang
# Github username: acse-ww721

import pandas as pd


def time_select(df, date_column, start_date, end_date):
    """
    Select rows from a DataFrame based on a date range.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_column (str): The name of the column containing dates.
        start_date (str): The start date of the range (in "YYYYMMDD" format).
        end_date (str): The end date of the range (in "YYYYMMDD" format).

    Returns:
        pd.DataFrame: A DataFrame containing rows within the specified date range.

    Example:
        >>> import pandas as pd
        >>> data = {'Date': ['20220101', '20220102', '20220103'],
        ...         'Value': [10, 15, 20]}
        >>> df = pd.DataFrame(data)
        >>> start_date = '20220102'
        >>> end_date = '20220103'
        >>> time_select(df, 'Date', start_date, end_date)
            Date  Value
        1  20220102     15
        2  20220103     20
    """
    df[date_column] = pd.to_datetime(df[date_column], format="%Y%m%d")
    filtered_df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
    return filtered_df


# Example
# filtered_noaa_df = time_select(processed_df_noaa, "DATE", start_date, end_date)
# filtered_asos_df = time_select(processed_df_asos, "date", start_date, end_date)


def is_leap_year(year):
    """
    Check if a year is a leap year.

    A leap year is divisible by 4, but not divisible by 100 unless it is also divisible by 400.

    Args:
        year (int): The year to be checked.

    Returns:
        bool: True if the year is a leap year, False otherwise.

    Example:
        >>> is_leap_year(2020)
        True
        >>> is_leap_year(2100)
        False
    """
    year = int(year)  # Convert the input to an integer
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def days_check(year, month):
    """
    Generate a list of days for a given year and month for ERA5 data format.

    Args:
        year (str): The year (as a string) for which to generate days.
        month (str): The month (as a string in "MM" format) for which to generate days.

    Returns:
        list: A list of day strings in "DD" format for the specified year and month.

    Example:
        >>> days_check("2023", "02")
        ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
    """
    year = int(year)  # Convert the input year to an integer
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
