# Name: Wenqi Wang
# Github username: acse-ww721

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def calculate_metrics(filename):
    # Load the data
    data = loadmat(filename)
    prediction = data["prediction"][:240]
    truth = data["truth"][:240]

    # Calculate Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((prediction - truth) ** 2, axis=(1, 2, 3)))

    # Calculate the correlation coefficient R
    reshaped_prediction = prediction.reshape(prediction.shape[0], -1)
    reshaped_truth = truth.reshape(truth.shape[0], -1)
    r_values = np.array(
        [
            np.corrcoef(reshaped_prediction[i], reshaped_truth[i])[0, 1]
            for i in range(reshaped_prediction.shape[0])
        ]
    )

    return rmse, r_values


def plot_combined_results(filename1, label1, filename2, label2):
    rmse1, r_values1 = calculate_metrics(filename1)
    rmse2, r_values2 = calculate_metrics(filename2)

    # Plotting
    time = np.arange(240)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # RMSE
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("RMSE")
    ax1.plot(time, rmse1, color="tab:red", label=label1)
    ax1.plot(time, rmse2, color="tab:orange", label=label2)
    for x in range(0, 240, 24):  # Add dashed lines every 24 hours
        ax1.axvline(x, color="gray", linestyle="--")
    ax1.legend()
    ax1.set_title("RMSE Over Time")

    # R value
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("R")
    ax2.plot(time, r_values1, color="tab:blue", label=label1)
    ax2.plot(time, r_values2, color="tab:cyan", label=label2)
    for x in range(0, 240, 24):  # Add dashed lines every 24 hours
        ax2.axvline(x, color="gray", linestyle="--")
    ax2.legend()
    ax2.set_title("Correlation Coefficient Over Time")

    fig.tight_layout()
    plt.show()


filename1 = "era5_DA_every24HR_lead1200_everytime_noise_" + str(1) + ".mat"
filename2 = "era5_DA_every24HR_lead1200_everytime_noise_" + str(0.5) + ".mat"

plot_combined_results(
    filename1,
    "U-STN12 + SPEnKF@24h with σ_obs = σ_T",
    filename2,
    "U-STN12 + SPEnKF@24h with σ_obs = 0.5 σ_T",
)
