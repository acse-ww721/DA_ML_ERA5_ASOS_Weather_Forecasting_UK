# Data Assimilation of ERA5 and ASOS with U-STN model for Weather Forecasting in the UK region

Author: Wenqi Wang

Contents
---------------------

<!-- TOC -->

1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Data](#data)
4. [Prediction Model](#prediction-model)
5. [Assimilation](#assimilation)
6. [Visualization](#visualization)
7. [Contributors](#contributors)
8. [License](#license)

<!-- TOC -->

Introduction
------------

In recent years, the integration of data-driven machine learning models with Data Assimilation (DA) has garnered
significant interest in enhancing model performance in weather forecasting. This study embarks on this trend, detailing
our approach and findings. We utilised the UK's local ERA5 T850 data and retrained the global weather forecasting model,
USTN12, to enhance its accuracy in predicting temperatures specific to the UK region. We acquired t2m data from the ASOS
ground observation stations across the UK. We applied the kriging method with polynomial drift term—an advanced
geostatistical procedure—for interpolation to achieve a uniform resolution. Additionally, based on the ERA5 T850 data,
Gaussian noise was randomly generated, laying the groundwork for subsequent multi-time step virtual observations. To
investigate the assimilation effects, we assimilated the ASOS t2m data into the ERA5 T850 data. Our results indicate
that while the original global forecast model can be migrated to cater to local regions, using atmospheric data for data
assimilation notably enhances model performance. However, assimilating surface temperature into atmospheric data
counters this improvement, diminishing the model's predictive capabilities.

Dependencies
------------
To establish the required environment for this project using Conda, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/acse-ww721/irp_ww721_bakcup.git
   cd irp_ww721_bakcup

To set up the required environment for this project, please follow the instructions below based on your operating
system:

1. Create a Conda virtual environment on Windows or Mac OS

   ```bash
   conda env create -f environment_win.yml
    ```
   or
    ```bash
    conda env create -f environment_mac.yml
     ```
2. Activate the virtual environment:
   ```bash
   conda activate your_environment_name
    ```
3. Install the required packages based on your operating system:

   ```bash
   pip install -r requirements_win.txt
   ```
   or
    ```bash
       conda install --file requirements_mac.txt
    ```

Data
------------

# Data Sources

This project relies on several primary data sources for its analysis:

1. **ERA5 Hourly Pressure Level Data (1940 - Present) from CDS**:
   - The project utilizes ERA5 hourly data on pressure levels ranging from 1940 to the present. This dataset can be
     accessed through the Copernicus Climate Data Store (CDS).
   - Data
     source: [CDS - ERA5 Pressure Levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview).

2. **ERA5 Hourly Single-Level Data (1940 - Present) from CDS**:
   - The project also leverages ERA5 hourly data on single levels covering the period from 1940 to the present,
     available through the Copernicus Climate Data Store (CDS).
   - Data
     source: [CDS - ERA5 Single Levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview).

3. **ASOS Hourly Observation Data (1979 - Present)**:
   - Hourly ASOS data, collected from 1979 to the present, is a vital component of this project. These observations are
     obtained from the Mesonet program and can be accessed for download.
   - Data source: [Mesonet ASOS Data](https://mesonet.agron.iastate.edu/request/download.phtml?network=GB__ASOS).

## Data Collection

The code available in the `src/data_collection` directory allows users to access, download, or crawl data from the
corresponding websites. While the primary focus is on the UK region, the code is designed to be adaptable for use in
other regions as well.

## Data Preprocessing

In the `src/data_preprocessing` folder, you will find detailed information on the specific preprocessing steps applied
to the data. These steps include handling missing data, interpolation, regridding, and data cleaning.

## Accessing Data

You can access various data sets related to this project, including raw data, processed data for training models, and
assimilated data, through the following Google Drive link:

[Google Drive - Project Data](https://drive.google.com/drive/folders/1JE6XWrNgVNdoxr4xXAQNjWPK_91YDCJM?usp=sharing)


Prediction Model
------------

# Model Implementation

The code for the model is available in the `src/model` directory. The model is implemented using Python and relies on
the Tensorflow library.

## Training and Validation

The model undergoes training and validation using ERA5 T850 data spanning from 1979 to 2020. For validation purposes,
ERA5 T850 data from the year 2021 is employed.

## Testing

The model's performance is rigorously evaluated through testing, utilizing ERA5 T850 data for the year 2022.

## Predictive Functionality

Subsequently, the model is deployed to predict temperature values at ASOS stations and ERA5 data points for a time
interval of 12 hours later.


Assimilation
------------

# Data Assimilation Using SPEnKF

For the assimilation of ASOS data, noisy model data, and virtual generated data into the ERA5 dataset, we employ the
Sigma Point Ensemble Kalman Filter (SPEnKF) technique.

## Implementation

The code responsible for the assimilation process can be found in the `src/assimilation` directory. This assimilation
procedure is implemented using Python.

## Inspirations

Our assimilation methodology draws inspiration from the work of [@ashesh6810](https://github.com/ashesh6810/DDWP-DA).
Their contributions have influenced the development of our assimilation approach.

Visualization
------------

You can access all the code related to the visualization part in the `src/visualization` directory. The code is
implemented using Python and relies on the Matplotlib library.

Contributors
------------
For any inquiries or issues with the code, please don't hesitate to reach out to me:

* [Wang, Wenqi](mailto:wenqi.wang21@imperial.ac.uk)

LICENSE
------------
This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.
