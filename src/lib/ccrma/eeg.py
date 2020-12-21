# -*- coding: utf-8 -*-

import scipy.io as sp
import numpy as np
import mne
from mne.io import read_raw_cnt
import matplotlib.pyplot as plt
from scipy import stats

# Method to load EEG topography montage from CCRMA Lab
# Returns the montage object
def loadMontage(mat_path):
    mat_channel_file = sp.loadmat(mat_path)['Channel']
    positions = np.zeros((len(mat_channel_file[0, :]), 3))
    
    for ichan in range(len(mat_channel_file[0, :])):
        positions[ichan, 0] = -mat_channel_file[0, ichan][3].T[0, 1]
        positions[ichan, 1] = mat_channel_file[0, ichan][3].T[0, 0]
        positions[ichan, 2] = mat_channel_file[0, ichan][3].T[0, 2]
    
    ch_names = [] # channel names are encoded in utf-8 (b'XXX), so they must be decoded.
    
    for ichan in range(len(mat_channel_file[0, :])):
        ch_names.append(str((mat_channel_file[0, ichan][0][0].encode('ascii')), "utf-8"))

    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, positions)), coord_frame =  'head')
    return montage

# Method to define object, raw
# Returns object CNT file
def loadRaw(fname, montage, eog_list, ecg_list,  misc_list):
    raw = mne.io.read_raw_cnt(fname, eog=eog_list, misc=misc_list, preload=True)
    stim_events = mne.find_events(raw, stim_channel='Trigger', output='step', shortest_event=1)
    return raw, stim_events

# Method to view raw CNT data
# Returns None
def viewRaw(raw, time):
    raw.set_channel_types(mapping={'HEO':'eog','VEO':'eog','Trigger':'misc'}) # Making both EOG channels be of 'eog' type.
    raw.plot(n_channels=1, duration=time, block=True)

# Method to apply SSP Projectors
# Returns modified CNT data
def applySSP(raw):
    # making HEO channel be of 'eog' type to find its projectors
    raw.set_channel_types(mapping={'HEO': 'eog', 'VEO': 'misc', 
        'Trigger': 'misc'})
    projs_heo, events = mne.preprocessing.compute_proj_eog(
        raw, n_eeg=1, average=True)  # SSP HEO
    # making VEO channel be of 'eog' type to find its projectors
    raw.set_channel_types(mapping={'HEO': 'misc', 'VEO': 'eog', 
        'Trigger': 'misc'})
    projs_veo, events = mne.preprocessing.compute_proj_eog(
        raw, n_eeg=1, average=True)  # SSP VEO
    
    if projs_heo != None:
        raw.add_proj(projs_heo)  # adding the projectors
    if projs_veo != None:
        raw.add_proj(projs_veo)  # adding the projectors
    
    raw.set_channel_types(mapping={'HEO':'eog','VEO':'eog','Trigger':'misc'})
    return raw

# Method to apply ICA Projectors
# Returns modified CNT data
def applyICA(raw):
    ica = ICA(n_components=25, method='fastica', random_state=23)
    print(ica)
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, eog=True,
                           stim=False, exclude='bads')
    ica.fit(raw, picks=picks_eeg, decim=3, reject=reject)
    print(ica)
    ica.plot_components()

# Method to Epoch
def epoch_data(raw, stim_events):
    event_id, tmin, tmax = {'Standard': 1}, -.2, 1.0
    epochs_params = dict(events=stim_events,
                        tmin=tmin, tmax=tmax, reject=None, proj=False, preload=True)
    epochs = mne.Epochs(raw, **epochs_params)

    # TODO
    # all_epochs = dict((cond, epochs[cond].get_data()) for cond in event_id) # stores epochs in an array, buggy.
    
    # visualization
    # epochs.plot(block=True, n_epochs=20, scalings=dict(eeg=70e-6))
    # epochs.average().plot(spatial_colors=True, time_unit='s')

# Method for autoreject
def apply_autoreject(raw, epochs):
    epochs = mne.Epochs(raw, **epochs_params)
    n_interpolates = np.array([1, 4, 8, 16])
    consensus_percs = np.linspace(0, 1.0, 11)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, include=[], exclude=[])
    ar = AutoReject(n_interpolates, consensus_percs, picks=picks, thresh_method='random_search', random_state=42)
    ar.fit(epochs)
    epochs_clean = ar.transform(epochs)
    epochs_clean = epochs_clean.apply_proj()
    
    # # TODO
    # # all_epochs = dict((cond, epochs_clean[cond].get_data()) for cond in event_id)
    
    # # visualization to compare to original epochs:
    # epochs_clean.plot(block=True,n_epochs=1,scalings=dict(eeg=70e-6))
    # epochs_clean.average().plot(spatial_colors=True, time_unit='s')


