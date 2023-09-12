# Name: Wenqi Wang
# Github username: acse-ww721

# The constant is calculated by The data from 1979-2020 (Training set)
# The calculation process can be seen in src/preprocessing/era5_preprocessing.py
# The running process can be seen in /src/models/test_Unet_STN_lead12.ipynb
# T850
MEAN_ERA5_1979_2020 = 273.77817
STD_ERA5_1979_2020 = 2.5819736

# The constant is calculated by The data from 1979-2022 ASOS regridded data R
# The calculation process can be seen in src/assimilation/test_calculation_t2m.py
# T2M
MEAN_T2M_2022 = 279.325
STD_T2M_2022 = 4.3148
