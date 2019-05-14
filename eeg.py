#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################
# author: Sebastian James  #
############################

'''
This is a Python pipeline which uses the MNE toolbox with dependencies
and the "autoreject" tool developed for EEG processing, to analyze
EEG data.

TODO:
- Pipline finished
- Change input to take in .gdf files
- Apply machine learning to epoched averages
'''

#%%
import mne
import os
from mne.io import read_raw_cnt
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp
from scipy import stats
from autoreject import get_rejection_threshold  # noqa
from autoreject import (AutoReject, set_matplotlib_defaults)  # noqa

# Will print system and MNE information, including all installed packages.
print("Current Working Directory ", os.getcwd())
print(mne.sys_info())
conditions = [1]

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

#%%
############################################################################
# Loop to: Load raw data, trim edges to provide only stream with triggers, #
# extract triggers, add any offset or filters needed, run SSP projections, #
# epoching, applying autoreject tools, and storing the data.               #
############################################################################

for icond in conditions: 

    raw_fname = 'Subj01_Data.cnt' # .cnt file must be in the working directory.
    raw = mne.io.read_raw_cnt(raw_fname, montage=montage, preload=True)
    stim_events = mne.find_events(raw, shortest_event=1)
    raw.crop(tmin=1, tmax=raw.times[-1] - 1)


    # OPTIONAL: visualize data to mark bad segments and bad electrodes
    raw.set_channel_types(mapping={'HEO':'eog','VEO':'eog','Trigger':'misc', 'STI 014':'misc'}) # Making both EOG channels be of 'eog' type.
    # raw.plot(n_channels=66,duration=20.0,block=True)
    print("BAD ELECTRODES") # Checking output
    print(raw.info['bads']) # Checking output
    print("ANNOTATIONS") # Checking output
    print(raw.annotations) # Checking output
    # # # interpolating bad channels
    # raw.interpolate_bads()
    # raw.info['bads'] = ['Trigger'] # Trigger is marked as bad as default. Let's keep it that way.
    
    # print(events)
    # print(raw.ch_names)
    # print(raw.info)

    
    ##################
    # SPP Projectors #
    ##################

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

    # Optional: view new data to check if projectors are correct. 
    raw.set_channel_types(mapping={'HEO':'eog','VEO':'eog','Trigger':'misc', 'STI 014':'misc'}) # making both EOG channels be of 'eog' type
    # raw.plot(n_channels=68,duration=20.0,block=True) # optional: visualize again to check the effect of SSP

    # Channels are now switched back to 'misc' for epoching, autoreject, and averaging.
    raw.set_channel_types(mapping={'HEO': 'misc', 'VEO': 'misc', 'Trigger': 'misc', 'STI 014': 'misc'})
    raw.interpolate_bads()

    #%%
    ############
    # Epoching #
    ############

    event_id, tmin, tmax = {'Standard': 1}, -.2, 1.0
    epochs_params = dict(events=stim_events,
                        tmin=tmin, tmax=tmax, reject=None, proj=False, preload=True)
    epochs = mne.Epochs(raw, **epochs_params)

    # TODO
    # all_epochs = dict((cond, epochs[cond].get_data()) for cond in event_id) # stores epochs in an array, buggy.
    
    # visualization
    # epochs.plot(block=True, n_epochs=20, scalings=dict(eeg=70e-6))
    # epochs.average().plot(spatial_colors=True, time_unit='s')

    
    ##############
    # Autoreject #
    ##############

    epochs = mne.Epochs(raw, **epochs_params)
    n_interpolates = np.array([1, 4, 8, 16])
    consensus_percs = np.linspace(0, 1.0, 11)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, include=[], exclude=[])
    ar = AutoReject(n_interpolates, consensus_percs, picks=picks, thresh_method='random_search', random_state=42)
    ar.fit(epochs)
    epochs_clean = ar.transform(epochs)
    epochs_clean = epochs_clean.apply_proj()
    
    # TODO
    # all_epochs = dict((cond, epochs_clean[cond].get_data()) for cond in event_id)
    
    # visualization to compare to original epochs:
    epochs_clean.plot(block=True,n_epochs=1,scalings=dict(eeg=70e-6))
    epochs_clean.average().plot(spatial_colors=True, time_unit='s')

    
    ##############
    # Store data #
    ##############

    evoked = np.concatenate((evoked, all_epochs['Standard']))
    print('Number of good trials: ', evoked.shape[0])


