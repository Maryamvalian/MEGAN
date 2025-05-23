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
# !jupytext --to py NCRF_volumesource.ipynb

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
meg=epochs.copy().pick_types(meg='mag')

# %%
len(time)

# %%
#TODO: ADD ALL predictors for all event types,set all values to 1. 
#Q:to make it both negative and positive (more similar to brain signals)  is'nt it beter to keep -1??
#  Make Stimulus Impulses 
stim1 = np.zeros(len(time), dtype=np.double)
stim1[events[:, 0]] = 1.

# -1 impulses to distinguish Word-real(real-inf or real-uninf) from pesudo  
stim2 = stim1.copy()
stim2[events[np.where(events[:, 2] == 162), 0]] = -1.
stim1 = eelbrain.NDVar(stim1, time)
stim2 = eelbrain.NDVar(stim2, time)

# visualize the stimulus
s = plot.LineStack(eelbrain.combine([stim1, stim2]))
# all 1 but for diferent predictor : 4 predictor

# %% [markdown]
# ## Forward Model

# %%
subject = "fsaverage"  # Use the standard FreeSurfer template
subjects_dir = "~/Data/Aphasia/mri/"
#Surface_Based:
#src = mne.setup_source_space(subject, spacing='ico4', add_dist=False, subjects_dir=subjects_dir)

#VOLUME BASED:
src = mne.setup_volume_source_space(
    subject=subject,
    pos=7.0,  # spacing in mm between grid points (adjustable, e.g., 5.0 or 10.0)
    mri='T1.mgz',  # typically 'T1.mgz' in the subject's MRI folder
    subjects_dir=subjects_dir
)
print(src)

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
print(trans)

#MAKE FORWARD MODEL

info=raw.info
fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=1)
print(fwd)
lf = fwd['sol']['data']
lf


# %% [markdown]
# ## Nois cov

# %%
evoked=meg["word/real"].average()
noise_cov = mne.make_ad_hoc_cov(evoked.info)
noise_cov

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
          'verbose': True, 'n_iter': 5, 'n_iterc': 10, 'n_iterf': 100}

model = fit_ncrf(*args, **kwargs)

# %%
#TODO:
#add all predictors to model
# Check ContinuousEpoch  on github repository 
#e.load_fwd(subject='') Use eelbrain pipeline to prepare parameters for ncrf
