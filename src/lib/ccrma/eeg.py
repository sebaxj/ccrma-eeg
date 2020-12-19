# -*- coding: utf-8 -*-

import scipy.io as sp
import numpy as np
import mne
from mne.io import read_raw_cnt
import matplotlib.pyplot as plt
from scipy import stats

# Method to load EEG topography montage from CCRMA Lab
# Returns the montage object
def loadMontage():
    mat_channel_file = sp.loadmat('channel_initial.mat')['Channel']
    positions = np.zeros((len(mat_channel_file[0, :]), 3))
    
    for ichan in range(len(mat_channel_file[0, :])):
        positions[ichan, 0] = -mat_channel_file[0, ichan][3].T[0, 1]
        positions[ichan, 1] = mat_channel_file[0, ichan][3].T[0, 0]
        positions[ichan, 2] = mat_channel_file[0, ichan][3].T[0, 2]
    
    ch_names = [] # channel names are encoded in utf-8 (b'XXX), so they must be decoded.
    
    for ichan in range(len(mat_channel_file[0, :])):
        ch_names.append(str((mat_channel_file[0, ichan][0][0].encode('ascii')), "utf-8"))

    montage = mne.channels.Montage(positions, ch_names, 'Neuroscan64', range(len(mat_channel_file[0, :])))
    return montage

# Method to define object, raw
# Returns object CNT file
def defineRaw(fname, montage):
    raw = mne.io.read_raw_cnt(fname, montage=montage, preload=False)
    stim_events = mne.find_events(raw, shortest_event=1)
    raw.set_channel_types(mapping={'HEO':'eog','VEO':'eog','Trigger':'misc', 'STI 014':'misc'}) # Making both EOG channels be of 'eog' type.
    return raw

# Method to view raw CNT data
# Returns None
def viewRaw(raw, time):
    raw.plot(n_channels=66, duration=time, block=True)

# Method to apply SSP Projectors
# Returns modified raw CNT data
def applySSP(raw):
    # making HEO channel be of 'eog' type to find its projectors
    raw.set_channel_types(mapping={'HEO': 'eog', 'VEO': 'misc', 
        'Trigger': 'misc', 'STI 014': 'misc'})
    projs_heo, events = mne.preprocessing.compute_proj_eog(
        raw, n_eeg=1, average=True)  # SSP HEO
    # making VEO channel be of 'eog' type to find its projectors
    raw.set_channel_types(mapping={'HEO': 'misc', 'VEO': 'eog', 
        'Trigger': 'misc', 'STI 014': 'misc'})
    projs_veo, events = mne.preprocessing.compute_proj_eog(
        raw, n_eeg=1, average=True)  # SSP VEO
    
    if projs_heo != None:
        raw.add_proj(projs_heo)  # adding the projectors
    if projs_veo != None:
        raw.add_proj(projs_veo)  # adding the projectors
    
    raw.set_channel_types(mapping={'HEO':'eog','VEO':'eog','Trigger':'misc', 'STI 014':'misc'})
    return raw

# Method to Epoch

# Method for autoreject
