from src.models.model_unet_stn_1x import stn
import scipy.io as sio
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Name: Wenqi Wang
# Github username: acse-ww721


def calculate_acc_and_rmse(era5_data, model_weight):
    """
    Calculate accuracy (correlation) and RMSE between ERA5 data and model predictions.

    Parameters:
        era5_data (str): Path to ERA5 dataset in NetCDF format.
        model_weight (str): Path to the pre-trained model weights.

    Returns:
        stepwise_acc_values (ndarray): Array of accuracy values for each time step.
        stepwise_rmse_values (ndarray): Array of RMSE values for each time step.

    This function loads a pre-trained model, ERA5 data, and calculates the accuracy (correlation)
    and RMSE between ERA5 data and model predictions. It randomly selects 50 initial conditions
    and predicts data for 24 hours. It then calculates accuracy and RMSE for each time step
    in the prediction.
    """

    # Load model
    model = stn()
    model.load_weights(model_weight)

    # Load data
    era5_value = xr.open_dataset(era5_data)
    era5_t = np.asarray(era5_value["t"])

    # Select 50 initial conditions randomly
    initial_conditions = 50
    random_indices = np.random.choice(
        era5_t.shape[0] - 240,
        size=initial_conditions,
        replace=False,  # -240 to ensure enough range for prediction
    )

    # Parameters
    dt = 24
    time_step = 240
    steps = time_step // dt

    pred_data = np.zeros([initial_conditions, steps, 32, 64])
    stepwise_acc_values = np.zeros(steps)  # Array to save ACC for each step
    stepwise_rmse_values = np.zeros(steps)  # Array to save RMSE for each step

    for idx, start_t in enumerate(random_indices):
        current_data = era5_t[start_t].reshape(1, 32, 64, 1)
        for s in range(steps):
            # Using the current data to predict the data for 24 hours later
            pred_data[idx, s] = model.predict(current_data).squeeze()
            # The predicted data is the current data for the next iteration
            current_data = pred_data[idx, s].reshape(1, 32, 64, 1)

    # Calculate accuracy and RMSE for each step
    for s in range(steps):
        acc_values_for_this_step = []
        rmse_values_for_this_step = []
        for i in range(initial_conditions):
            actual_data = era5_t[random_indices[i] + s * dt]
            predicted_data = pred_data[i, s]

            correlation, _ = pearsonr(actual_data.flatten(), predicted_data.flatten())
            rmse = np.sqrt(mean_squared_error(actual_data, predicted_data))

            acc_values_for_this_step.append(correlation)
            rmse_values_for_this_step.append(rmse)

        stepwise_acc_values[s] = np.mean(acc_values_for_this_step)
        stepwise_rmse_values[s] = np.mean(rmse_values_for_this_step)

    return stepwise_acc_values, stepwise_rmse_values
