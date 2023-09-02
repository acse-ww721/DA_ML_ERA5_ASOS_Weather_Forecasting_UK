from src.models.model_unet_stn_1x import stn
import scipy.io as sio
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm


def calculate_acc(era5_data, stn_model_data, model_weight):
    # acc_value = 1 - np.mean(np.abs(input_value))

    # Load model
    model = stn()
    model.load_weights(model_weight)

    random_seed = 42
    np.random.seed(random_seed)

    # Set time step
    time_step = 240
    dt = 24
    steps = time_step // dt
    count = 0

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

    pred_data = np.zeros([initial_conditions, steps, 32, 64])

    for idx, start_t in tqdm(enumerate(random_indices)):
        current_data = era5_t[start_t].reshape(1, 32, 64, 1)
        for s in range(steps):
            # Using the current data to predict the data for 24 hours later
            pred_data[idx, s] = model.predict(current_data).squeeze()
            # The predicted data is the current data for the next iteration
            current_data = pred_data[idx, s].reshape(1, 32, 64, 1)

    # Calculate accuracy
    stepwise_acc_values = np.zeros(steps)
    # Calculate accuracy for each step
    for s in range(steps):
        acc_values_for_this_step = []
        for i in range(initial_conditions):
            actual_data = era5_t[random_indices[i] + s*dt]
            predicted_data = pred_data[i, s]

            correlation, _ = pearsonr(actual_data.flatten(), predicted_data.flatten())
            acc_values_for_this_step.append(correlation)
        stepwise_acc_values[s] = np.mean(acc_values_for_this_step)

    return stepwise_acc_values