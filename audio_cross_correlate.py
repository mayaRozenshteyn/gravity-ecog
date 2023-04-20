import os
import sys
import numpy as np
from scipy.signal import correlate, correlation_lags

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

def cross_correlate(film_data, neural_data):
    xcorr_arr = np.zeros((neural_data.shape[0], film_data.shape[0] + neural_data.shape[1] - 1))
    lags_arr = np.zeros((neural_data.shape[0], film_data.shape[0] + neural_data.shape[1] - 1))
    for i in range(neural_data.shape[0]):
        xcorr_arr[i,:] = correlate(film_data - np.mean(film_data), neural_data[i, :] - np.mean(neural_data[i, :]), mode='full')
        lags_arr[i,:] = correlation_lags(film_data.shape[0], neural_data.shape[1], mode="full")
    return xcorr_arr, lags_arr

shuffled_signals = np.load(os.path.join(sys.path[0], "shuffled_audio_g1.npy"), allow_pickle = True)
neural_data = np.load(os.path.join(sys.path[0], "# Removed to protect  patient anonymity"), allow_pickle = True)

xcorr_arr, lags_arr = cross_correlate(shuffled_signals[idx, :], neural_data)
zero_lag_idx = np.where(lags_arr[0, :] == 0)[0][0]

corr = xcorr_arr[:, zero_lag_idx - 35*512:zero_lag_idx + 35*512 + 1]

np.save(os.path.join(sys.path[0], "# Removed to protect  patient anonymity"), corr)