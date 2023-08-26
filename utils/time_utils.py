import pandas as pd


def time_select(df, date_column, start_date, end_date):
    df[date_column] = pd.to_datetime(df[date_column], format="%Y%m%d")
    filtered_df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
    return filtered_df


# Example
# filtered_noaa_df = time_select(processed_df_noaa, "DATE", start_date, end_date)
# filtered_asos_df = time_select(processed_df_asos, "date", start_date, end_date)


def is_leap_year(year):
    year = int(year)  # Convert the string to integer
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def days_check(year, month):
    year = int(year)  # Convert the string to integer
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
