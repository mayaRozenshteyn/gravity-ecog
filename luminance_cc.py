import os
import sys
import numpy as np
from scipy.signal import correlate, correlation_lags
import pickle

id_eleccount = # Removed to protect patient anonymity
vis_elec = pickle.load(open(os.path.join(sys.path[0], "../../sourcedata/vis_elec_idx.pkl"), "rb"))

def cross_correlate(film_data, neural_data):
    xcorr_arr = np.zeros(film_data.shape[0] + neural_data.shape[0] - 1)
    lags_arr = np.zeros(film_data.shape[0] + neural_data.shape[0] - 1)
    xcorr_arr = correlate(film_data - np.mean(film_data), neural_data - np.mean(neural_data), mode='full')
    lags_arr = correlation_lags(film_data.shape[0], neural_data.shape[0], mode="full")
    return xcorr_arr, lags_arr

def extract_cc(patient_id):
    sf_1 = np.load(os.path.join(sys.path[0], "tr_shuffled_luminance_g1.npy"), allow_pickle = True)
    sf_2 = np.load(os.path.join(sys.path[0], "tr_shuffled_luminance_g2.npy"), allow_pickle = True)
    
    nr_1 = np.load(os.path.join(sys.path[0], "../../sourcedata/tie-ranked-ecog/ecog1-" + patient_id + ".npy"), allow_pickle = True)
    nr_2 = np.load(os.path.join(sys.path[0], "../../sourcedata/tie-ranked-ecog/ecog2-" + patient_id + ".npy"), allow_pickle = True)
    
    for channel in vis_elec[patient_id]:
        full_cc_1 = np.zeros((1000, 35*512*2+1))
        for i in range(1000):
            cc_1, l_1 = cross_correlate(sf_1[i, :], nr_1[channel, :])
            zero_lag_idx_1 = np.where(l_1 == 0)[0][0]
            full_cc_1[i, :] = cc_1[zero_lag_idx_1 - 35*512:zero_lag_idx_1 + 35*512 + 1]

        np.save(os.path.join(sys.path[0], "tr-cc1-" + patient_id + "-" + str(channel) + ".npy"), full_cc_1)
        
    for channel in vis_elec[patient_id]:
        full_cc_2 = np.zeros((1000, 35*512*2+1))
        for i in range(1000):
            cc_2, l_2 = cross_correlate(sf_2[i, :], nr_2[channel, :])
            zero_lag_idx_2 = np.where(l_2 == 0)[0][0]
            full_cc_2[i, :] = cc_2[zero_lag_idx_2 - 35*512:zero_lag_idx_2 + 35*512 + 1]
        
        np.save(os.path.join(sys.path[0], "tr-cc2-" + patient_id + "-" + str(channel) + ".npy"), full_cc_2)
    

for key in id_eleccount.keys():
    extract_cc(key)