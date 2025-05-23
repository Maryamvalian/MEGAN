# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
import eelbrain 
from matplotlib import pyplot
import mne
import numpy as np
import pickle
from ncrf import fit_ncrf
from eelbrain import load, plot
# %run experiment.py
# e is the experiment



# %%
e.set(raw='1-8',epoch='word')
raw = e.load_raw(ndvar=False)  #ndvar true doesnt have Resample
print(f"Current sample rate= {raw.info['sfreq']}")
# resample to 100 Hz to reduce memory
raw.resample(100, npad="auto")
#raw.plot()

# %%
time_size = raw.n_times / raw.info['sfreq']
print("Duration:", time_size, "seconds")

# %%
session='words'
group='young'
epochs_list= ['word-real','word-stem','word-inflected']

time = eelbrain.UTS.from_int(0, raw.n_times-1, raw.info['sfreq'])
len(time)


# %%
#events= e.load_events()
#events

event_dict = {
     "word/real":162,
     "word/stem":163,
    "word/uninf":164,
    "word/inf":165
    }
   

# %%
#e.make_epoch_selection()

# %%
#epochs=e.load_epochs(epoch='word-stem',ndvar=True) 
#epochs
events = mne.find_events(raw, stim_channel='STI 014')
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.7, preload=True)

picks = mne.pick_types(epochs.info, meg='mag', eeg=False, stim=False, eog=False, exclude='bads')
data = epochs.get_data()[:, picks, :]  # shape = (n_trials, n_sensors, n_times)

# Create Eelbrain NDVar
sensor_dim = load.fiff.sensor_dim(epochs.info, picks=picks)
time = eelbrain.UTS(epochs.tmin, 1 / epochs.info['sfreq'], data.shape[2])
case_dim = eelbrain.Case(data.shape[0])
meg = eelbrain.NDVar(data, dims=(case_dim, sensor_dim, time))

# %%
len(time)

# %%
#TODO: ADD ALL predictors for all event types,set all values to 1. 
#Q:to make it both negative and positive (more similar to brain signals)  is'nt it beter to keep -1??
#  Make Stimulus Impulses 

event_ids = epochs.events[:, 2]
n_trials, n_times = data.shape[0], data.shape[2]


stim1_array = np.zeros((n_trials, n_times))
stim2_array = np.zeros((n_trials, n_times))
stim3_array = np.zeros((n_trials, n_times))


#stim1_array[:, 0] = 1.  
stim1_array[event_ids == 162, 0] = 1.           #real
stim2_array[event_ids == 163, 0] = 1.           #stem
stim3_array[event_ids == 164, 0] = 1.           #uinflected
stim4_array[event_ids == 165, 0] = 1.           #inf


stim1 = eelbrain.NDVar(stim1_array, dims=(case_dim, time))
stim2 = eelbrain.NDVar(stim2_array, dims=(case_dim, time))
stim3 = eelbrain.NDVar(stim1_array, dims=(case_dim, time))
stim4 = eelbrain.NDVar(stim1_array, dims=(case_dim, time))
# visualize the stimulus
#s = plot.LineStack(eelbrain.combine([stim1, stim2]))
# all 1 but for diferent predictor : 4 predictor

# %% [markdown]
# ## Forward Model

# %%
import os
subjects_dir = os.path.expanduser('~/Data/Aphasia/mri')
subject = "fsaverage"  
#Surface_Based:
#src = mne.setup_source_space(subject, spacing='ico4', add_dist=False, subjects_dir=subjects_dir)

#VOLUME BASED:
src = mne.setup_volume_source_space(
    subject=subject,
    pos=7.0,  # vol-7 ,vol-5 or vol-10
    mri='T1.mgz',  
    subjects_dir=subjects_dir
)
print(src)

#save Source 
mne.write_source_spaces(
    os.path.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-vol-7-src.fif'),
    src,
    overwrite=True
)
#BEM

conductivity = (0.3, 0.006, 0.3)  # for three layers
#define geometry of head and conductivity

#BEM IS SAME for both Volume and surface!
BEM_model = mne.make_bem_model(
    subject="fsaverage", ico=4, conductivity=conductivity, subjects_dir=subjects_dir
)
#solve Maxwell equations for the model
bem = mne.make_bem_solution(BEM_model)

#CO-registration

trans = mne.read_trans("~/Data/Aphasia/meg/R0240/R0240-trans.fif")
print(f"TRANS={trans}")






# %%

#MAKE FORWARD MODEL
info=raw.info
fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=1)
print(fwd)
#lf = fwd['sol']['data']

lf = eelbrain.load.fiff.forward_operator(
    fwd,  
    src='vol-7',  # or 'ico-4' 
    subjects_dir=subjects_dir,
    sysname='neuromag306mag',
    connectivity=False,              #avoid mismatch chnl name
    parc='aparc+aseg'
)


# %% [markdown]
# ## Nois cov

# %%
# empty-room data
raw_empty = mne.io.read_raw_fif('~/Data/Aphasia/meg/R0240/R0240_emptyroom-raw.fif', preload=True)


raw_empty.notch_filter(np.arange(60, 181, 60), fir_design='firwin')
raw_empty.filter(1., 8., fir_design='firwin')
raw_empty.resample(100, npad="auto")


noise_cov = mne.compute_raw_covariance(raw_empty, method='shrunk', rank=None)



# %% [markdown]
# ## NCRF Estimation

# %%
# %load_ext autoreload
# %autoreload 2

import eelbrain
from ncrf import fit_ncrf

args=(meg, [stim1, stim2], lf, noise_cov, 0, 0.8)

mu = 0.0002
kwargs = {'normalize': 'l1', 'in_place': False, 'mu': mu,
          'verbose': True, 'n_iter': 5, 'n_iterc': 10, 'n_iterf': 50}   #n_iterf=100

model = fit_ncrf(*args, **kwargs)

# %%
eelbrain.save.pickle(model, 'model_2sim_n100.pickle')

# %%
#TODO:
#add all predictors to model
# Check ContinuousEpoch  on github repository 
#e.load_fwd(subject='') Use eelbrain pipeline to prepare parameters for ncrf
