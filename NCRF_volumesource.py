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
import numpy
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

time = eelbrain.UTS.from_int(0, time_size.size-1, raw.info['sfreq'])

# %%
events= e.load_events()
#events

# %%
e.make_epoch_selection()

# %%
epochs=e.load_epochs(ndvar=True) 
#epochs

# %%
