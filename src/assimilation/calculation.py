from src.models.model_unet_stn_1x import stn
import scipy.io as sio
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def calculate_acc(era5_data, stn_model_data, model_weight):
    # acc_value = 1 - np.mean(np.abs(input_value))

    # Load model
    model = stn()
    model.load_weights(model_weight)

    random_seed = 42
    np.random.seed(random_seed)

    # Load data

    era5_value = xr.open_dataset(era5_data)
    era5_t = np.asarray(era5_value["t"])

    stn_value = xr.open_dataset(stn_model_data)
    stn_t = np.asarray(stn_value["t"])

    # Select 50 initial conditions randomly
    initial_conditions = 50
    random_indices = np.random.choice(
        range(era5_data.shape[0]), size=initial_conditions, replace=False
    )

    # Set time step
    time_step = 240
    dt = 24
    count = 0
    pred_era5 = np.zeros([time_step, 32, 64, 1])
    pred_stn = np.zeros([time_step, 32, 64, 1])

    for t in range(0, time_step, dt):
        for kk in range(0, dt - 1):
            if kk == 0:
                random_era5_t = era5_t[t + kk, :, :].reshape([1, 32, 64, 1])
                random_era5_t = model.predict(random_era5_t.reshape([1, 32, 64, 1]))

                random_stn_t = stn_t[t + kk, :, :].reshape([1, 32, 64, 1])
                random_stn_t = model.predict(random_stn_t.reshape([1, 32, 64, 1]))

            else:
                random_era5_t = model.predict(random_era5_t)
                random_stn_t = model.predict(random_stn_t)

        pred_era5[count, :, :, 0] = np.reshape(random_era5_t, [32, 64])
        pred_stn[count, :, :, 0] = np.reshape(random_stn_t, [32, 64])
        count = count + 1

    # Calculate accuracy
    acc_values = []
    for i in range(initial_conditions):
        era5_anomaly = (
            era5_t[random_indices[i] : random_indices[i + 240], :, :]
            - pred_era5[i, :, :, 0]
        )
        stn_anomaly = (
            stn_t[random_indices[i] :: random_indices[i + 240], :, :]
            - pred_stn[i, :, :, 0]
        )

        correlation, _ = pearsonr(era5_anomaly.flatten(), stn_anomaly.flatten())
        acc_values.append(correlation)

    # for random_index in random_indices:
    #     # Select random initial condition
    #     random_era5_t = era5_t[random_index, :, :]
    #     random_stn_t = stn_t[random_index, :, :]
    #
    #     era5_pred = random_era5_t.reshape([1, 32, 64, 1])
    #     era5_pred = model.predict(era5_pred.reshape([1, 32, 64, 1]))
    #
    #     stn_pred = random_stn_t.reshape([1, 32, 64, 1])
    #     stn_pred = model.predict(stn_pred.reshape([1, 32, 64, 1]))


sio.savemat(
    "DA_every24HR_lead1200_everytime_noise_" + str(noise) + ".mat",
    dict(
        [
            ("prediction", pred),
            ("truth", np.reshape(TRUTH, [np.size(Z_rs, 0), 32, 64, 1])),
            ("noisy_obs", np.reshape(Z_rs, [np.size(Z_rs, 0), 32, 64, 1])),
        ]
    ),
)

print("Done writing file")
