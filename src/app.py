#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################
# author: Sebastian James  #
############################

'''
This is a Python pipeline which uses the MNE toolbox with dependencies
and the "autoreject" tool developed for EEG processing, to analyze
EEG data.

Pipeline:
    - Load Montage
    - Load Raw
    - View Raw
    - View artifacts
    - Apply pre-processing (SSP or ICA)
    - Apply autoreject
    - Epoch data
    - Average data
    - View Grand Average

TODO:

'''
import mne
import os
from lib.ccrma import eeg as e
from mne.io import read_raw_cnt
from mne.preprocessing import create_eog_epochs, compute_proj_eog
from mne.preprocessing import ICA
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
reject=dict(eeg=1e-4)

# Setting up the general montage for the Neuroscan64 Quik-Cap
# used in the CCRMA Neuromusic Lab.

montage  = e.loadMontage('include/channel_initial.mat')

############################################################################
# Loop to: Load raw data, trim edges to provide only stream with triggers, #
# extract triggers, add any offset or filters needed, run SSP projections, #
# epoching, applying autoreject tools, and storing the data.               #
############################################################################

raw_fname = '../Subj01_Data.cnt' # .cnt file must be in the working directory.

for icond in conditions: 
    raw, stim_events = e.loadRaw(raw_fname, montage, ['HEO', 'VEO'], [], ['Trigger'])
    raw.crop(tmin=1, tmax=raw.times[-1] - 1)

    # OPTIONAL: visualize data to mark bad segments and bad electrodes
    e.viewRaw(raw, 20.0)

    # Check back electrodes
    print("BAD ELECTRODES") # Checking output
    print(raw.info['bads']) # Checking output
    print("ANNOTATIONS") # Checking output
    print(raw.annotations) # Checking output
    raw.info['bads'] = ['Trigger'] # Trigger is marked as bad as default. Let's keep it that way.
    print(raw.info)

    # Check stim events
    print(stim_events)

    ##################
    # SPP Projectors #
    ##################

    # Check epochs around eye blinks usinng EOG channel
    average_eog = create_eog_epochs(raw).average()
    print('We found %i EOG events' % average_eog.nave)
    average_eog.plot_joint()

    # apply SSP
    e.applySSP(raw)
    
    # Optional: view new data to check if projectors are correct. 
    raw.set_channel_types(mapping={'HEO':'eog','VEO':'eog','Trigger':'misc'}) # making both EOG channels be of 'eog' type
    raw.plot(n_channels=1,duration=20.0,block=True) # optional: visualize again to check the effect of SSP

    # Channels are now switched back to 'misc' for epoching, autoreject, and averaging.
    raw.set_channel_types(mapping={'HEO': 'misc', 'VEO': 'misc', 'Trigger': 'misc'})

    
    # evoked = np.concatenate((evoked, all_epochs['Standard']))
    # print('Number of good trials: ', evoked.shape[0])

