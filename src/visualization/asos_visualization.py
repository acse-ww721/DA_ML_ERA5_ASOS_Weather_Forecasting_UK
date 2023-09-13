# Name: Wenqi Wang
# Github username: acse-ww721

# Being used in the test_data_asos_regrid_and_plot.ipynb

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_uk_with_temperatures(
    temperature_mean, latitudes, longitudes, vmin=282, vmax=290
):
    """
    Plot a map of the United Kingdom with temperature data.

    Parameters:
        temperature_mean (array-like): An array of temperature mean values.
        latitudes (array-like): An array of latitude coordinates.
        longitudes (array-like): An array of longitude coordinates.
        vmin (float, optional): Minimum value for color mapping. Default is 282.
        vmax (float, optional): Maximum value for color mapping. Default is 290.

    Returns:
        None

    This function creates a map plot of temperature data for the United Kingdom
    using Matplotlib and Cartopy. It displays temperature values as a scatter plot
    on the map, with the option to customize the color mapping range.
    """
    # Create a Matplotlib figure and axes, using the PlateCarree projection
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

    # Set the geographical extent to cover the surroundings of the United Kingdom
    ax.set_extent([-10, 3, 49, 60], crs=ccrs.PlateCarree())

    # Add the border of the United Kingdom
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    # Add coastlines and lakes
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAKES, edgecolor="black")

    # Add gridlines with labels
    ax.gridlines(draw_labels=True)

    # Set the labels for the x-axis and y-axis
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Scatter plot temperature data on the map
    scatter_size = 50  # Modify this value to adjust point size
    plt.scatter(
        longitudes,
        latitudes,
        c=temperature_mean,
        s=scatter_size,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )

    # Add a colorbar to indicate temperature values
    plt.colorbar(label="Temperature Mean (C)")

    # Show the map
    plt.show()


# Assuming latitudes and longitudes are arrays of station latitudes and longitudes
# plot_uk_with_temperatures(
#     temperature_mean_by_station["t2m"],
#     temperature_mean_by_station["latitude"],
#     temperature_mean_by_station["longitude"],
# )
