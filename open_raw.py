#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Script to look at Raw CNT data collected with the Neuroscan QuickCap 64
and view events, epochs, and artifacts.
'''
#%%
import os
import mne
from mne.io import read_raw_cnt
import scipy.io as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Will print system and MNE information, including all installed packages.
print("Current Working Directory ", os.getcwd())
print(mne.sys_info())

# Setting up the general montage for the Neuroscan64 Quik-Cap
# used in the CCRMA Neuromusic Lab.

mat_channel_file = sp.loadmat('channel_initial.mat')['Channel'] # .mat file must be in working directory.
positions = np.zeros((len(mat_channel_file[0, :]), 3))
for ichan in range(len(mat_channel_file[0, :])):
    positions[ichan, 0] = -mat_channel_file[0, ichan][3].T[0, 1]
    positions[ichan, 1] = mat_channel_file[0, ichan][3].T[0, 0]
    positions[ichan, 2] = mat_channel_file[0, ichan][3].T[0, 2]
ch_names = [] # channel names are encoded in utf-8 (b'XXX), so they must be decoded.
for ichan in range(len(mat_channel_file[0, :])):
    ch_names.append(str((mat_channel_file[0, ichan][0][0].encode('ascii')), "utf-8"))

montage = mne.channels.Montage(positions, ch_names, 'Neuroscan64', range(len(mat_channel_file[0, :])))

# Opening raw
raw_fname = 'Subj01_Data.cnt' # .cnt file must be in the working directory.
raw = mne.io.read_raw_cnt(raw_fname, montage=montage, preload=True)
stim_events = mne.find_events(raw, shortest_event=1)
raw.set_channel_types(mapping={'HEO':'eog','VEO':'eog','Trigger':'misc', 'STI 014':'misc'}) # Making both EOG channels be of 'eog' type.
raw.crop(tmin=18.0, tmax=raw.times[-1] - 5.8)
raw.plot(n_channels=66,duration=1.0,block=True)


# Add epochs and events to view
event_id, tmin, tmax = {'Standard': 1}, -.2, 1.0
epochs_params = dict(events=stim_events,
                    tmin=tmin, tmax=tmax, reject=None, proj=False, preload=True)
epochs = mne.Epochs(raw, **epochs_params)
epochs.plot(block=True, n_epochs=20, scalings=dict(eeg=70e-6))