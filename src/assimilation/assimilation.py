import os
import numpy as np
from collections import defaultdict
from enum import Enum
from openda.utils import Matrix, Results
from openda.utils.io import FileBasedModelState, KalmanGainStorage
from openda.interfaces import (
    IVector,
    IMatrix,
    ITime,
    IResultWriter,
    IObservationDescriptions,
    IStochObserver,
    IStochModelInstance,
)


class LocalizationMethodType(Enum):
    hamill = 1
    autoZhang = 2


class EnKF:
    def __init__(self):
        self.gainVectors = defaultdict(IVector)
        self.obsId = {}
        self.obsTimeOffset = {}
        self.smoothedGainMatrix = None
        # Add other attributes here

    def analysis(
        self,
        obs: IStochObserver,
        obsValues: IVector,
        predictions: IVector,
        mainModel: IStochModelInstance,
        analysisTime: ITime,
    ):
        self.init_all_timers()

        with self.timer_total:
            ensemble_vectors_forecast = self.get_ensemble_vectors_state(False)
            ensemble_predictions_forecast = self.get_ensemble_vectors_prediction(
                obs.get_observation_descriptions(), False
            )

            self.set_current_state(ensemble_vectors_forecast.mean.clone())

            Kvecs = self.compute_gain_matrix(
                obs, ensemble_predictions_forecast, ensemble_vectors_forecast
            )

            self.apply_localization_to_gain(
                obs, Kvecs, ensemble_predictions_forecast, ensemble_vectors_forecast
            )

            if self.time_regularisation_per_day > 0.0:
                if self.smoothed_gain_matrix is None:
                    self.smoothed_gain_matrix = SmoothedGainMatrix(
                        self.time_regularisation_per_day
                    )
                self.smoothed_gain_matrix.smooth_gain(
                    obs, Kvecs, self.time_regularisation_per_day, analysisTime
                )

            self.store_gain_matrix(obs, analysisTime, Kvecs)

            self.update_model_with_gain(
                obs, ensemble_predictions_forecast, ensemble_vectors_forecast, Kvecs
            )

            for j in range(len(Kvecs)):
                Kvecs[j].free()
            ensemble_vectors_forecast.free()
            ensemble_predictions_forecast.free()

            ensemble_vectors_analysis = self.get_ensemble_vectors_state(True)
            ensemble_predictions_analysis = self.get_ensemble_vectors_prediction(
                obs.get_observation_descriptions(), True
            )

            self.update_main_model(ensemble_vectors_analysis)

            ensemble_vectors_analysis.free()
            ensemble_predictions_analysis.free()

    def init_all_timers(self):
        if self.timer_total is None:
            self.timer_total = OdaTiming("analysis")
            self.timer_synchronize = OdaTiming("Synchronize")
            # Add other timers here
            self.timer_total.add_sub_timer(self.timer_synchronize)
            # Add other sub-timers here

    # Add other methods here


class SmoothedGainMatrix:
    def __init__(self, time_regularisation_per_day):
        self.last_gain_matrix_hash_map = {}
        self.obs_ids = {}
        self.last_obs_time_offsets = {}
        self.previous_analysis_time = None
        self.time_regularisation_per_day = time_regularisation_per_day

    def smooth_gain(self, obs, K, time_regularisation_per_day, analysis_time):
        n_cols = len(K)
        obs_utils = ObserverUtils(obs)
        obs_ids = obs_utils.get_obs_ids()
        obs_time_offsets = obs_utils.get_obs_time_offsets(analysis_time.get_mjd())

        if not self.last_gain_matrix_hash_map:
            for i in range(obs.get_count()):
                gain_vector_id = (
                    f"{obs_ids[i]}:{round(obs_time_offsets[i] * 24.0 * 3600.0)}"
                )
                self.last_gain_matrix_hash_map[gain_vector_id] = K[i].clone()
                self.obs_ids[gain_vector_id] = obs_ids[i]
                self.last_obs_time_offsets[gain_vector_id] = obs_time_offsets[i]
        else:
            # TODO: Implement the rest of the smoothing logic here
            pass


# Instantiate EnKF and call the analysis method
enkf = EnKF()
# Initialize other variables and objects as needed
enkf.analysis(obs, obs_values, predictions, main_model, analysis_time)
