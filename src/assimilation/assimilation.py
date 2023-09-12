# The function implementation below is a modification version from Tensorflow
# Original code link: https://github.com/ashesh6810/DDWP-DA/blob/master/EnKF_DD_all_time.py

import numpy as np
import netCDF4 as nc
import scipy.io as sio
from src.models.model_unet_stn_1x import stn

### This .mat file has been generated from the ERA5 lat-lon data ####
file = sio.loadmat("ERA_grid.mat")
lat = file["lat"]
lon = file["lon"]

########## This is the testing set #######

fileList_test = []
fileList_test.append("geopotential_500hPa_2018_5.625deg.nc")


########### Ensure same normalization coefficient as trainig #######
file = nc.Dataset("ERA_Z500_1hour.nc")
Z500 = np.asarray(file["input"])
M = np.mean(Z500.flatten())
sdev = np.std(Z500.flatten())


####### True data (noise free) for twin DA experiments ##########
F = nc.Dataset(fileList_test[0])
Z = np.asarray(F["z"])
TRUTH = Z

### Meshgrid for plotting ###
[qx, qy] = np.meshgrid(lon, lat)


##### Add noise to the truth to mimic observations####
#### Value 1 is 1*\sigma_Z. See more in paper #####
Z_rs = np.reshape(Z, [np.size(Z, 0), int(np.size(Z, 1) * np.size(Z, 2))])
TRUTH = Z_rs
Z_rs = (Z_rs - M) / sdev
TRUTH = (TRUTH - M) / sdev
noise = 1
for k in range(1, np.size(Z_rs, 0)):
    Z_rs[k - 1, :] = Z_rs[k - 1, :] + np.random.normal(0, noise, 2048)


print("length of initial condition", len(Z_rs[0, :]))

#### SPNEKF implementation following Tyrus Berry's implementation ######


def ENKF(x, n, P, Q, R, obs, model, u_ensemble):
    """
    Perform Ensemble Kalman Filter (EnKF) data assimilation.

    Parameters:
        x (ndarray): The state vector to be updated.
        n (int): The size of the state vector.
        P (ndarray): The covariance matrix of the state vector.
        Q (ndarray): The process noise covariance matrix.
        R (ndarray): The observation noise covariance matrix.
        obs (ndarray): The observed data.
        model: The predictive model used for forecasting.
        u_ensemble (ndarray): An array to store the ensemble of forecasts.

    Returns:
        x_updated (ndarray): The updated state vector.
        P_updated (ndarray): The updated covariance matrix.

    This function performs the Ensemble Kalman Filter (EnKF) data assimilation to update
    the state vector 'x' and its covariance matrix 'P' based on observations 'obs'.
    The process noise covariance 'Q' and observation noise covariance 'R' are also used.
    The predictive model 'model' is used for forecasting, and the ensemble of forecasts
    is stored in 'u_ensemble'.
    """

    obs = np.reshape(obs, [n, 1])
    x = np.reshape(x, [n, 1])
    [U, S, V] = np.linalg.svd(P)
    D = np.zeros([n, n])
    np.fill_diagonal(D, S)
    sqrtP = np.dot(np.dot(U, np.sqrt(D)), U)
    ens = np.zeros([n, 2 * n])
    ens[:, 0:n] = np.tile(x, (1, n)) + sqrtP
    ens[:, n:] = np.tile(x, (1, n)) - sqrtP
    ## forecasting step,dummy model

    for k in range(0, np.size(ens, 1)):
        u = model.predict(np.reshape(ens[:, k], [1, 32, 64, 1]))

        u_ensemble[:, k] = np.reshape(u, (32 * 64,))

    ############################
    x_prior = np.reshape(np.mean(u_ensemble, 1), [n, 1])
    print("shape pf x_prior", np.shape(x_prior))
    print("shape pf obs", np.shape(obs))
    cf_ens = ens - np.tile(x_prior, (1, 2 * n))
    P_prior = np.dot(cf_ens, np.transpose(cf_ens)) / (2 * n - 1) + Q
    h_ens = ens
    y_prior = np.reshape(np.mean(h_ens, 1), [n, 1])
    ch_ens = h_ens - np.tile(y_prior, (1, 2 * n))
    print("shape pf y_prior", np.shape(y_prior))
    P_y = np.dot(ch_ens, np.transpose(ch_ens)) / (2 * n - 1) + R
    P_xy = np.dot(cf_ens, np.transpose(ch_ens)) / (2 * n - 1)
    K = np.dot(P_xy, np.linalg.inv(P_y))
    P = P_prior - np.dot(np.dot(K, P_y), np.transpose(K))
    x = x_prior + np.dot(K, (obs - y_prior))

    return x, P


model = stn()
model.load_weights("best_weights_lead1.h5")
### This code performs DA at every 24 hrs with a model that is forecasting every hour. So lead will always be 1 ######


###### Start Data Assimilation Process #########################################

time = 1200
n = int(32 * 64)
P = np.eye(n, n)

Q = 0.03 * np.eye(n, n)

R = 0.0001

u_ensemble = np.zeros([32 * 64, 2 * 32 * 64])

pred = np.zeros([time, 32, 64, 1])


dt = 24
count = 0
for t in range(0, time, dt):
    for kk in range(0, dt - 1):
        if kk == 0:
            u = Z_rs[t + kk, :].reshape([1, 32, 64, 1])
            u = model.predict(u.reshape([1, 32, 64, 1]))
        else:
            u = model.predict(u)

        pred[count, :, :, 0] = np.reshape(u, [32, 64])
        count = count + 1
    x = u
    x, P = ENKF(x, 2048, P, Q, R, Z_rs[t + dt, :], model, u_ensemble)

    print("output shape of ENKF", np.shape(x))

    pred[count, :, :, 0] = np.reshape(x, [32, 64])
    count = count + 1


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
