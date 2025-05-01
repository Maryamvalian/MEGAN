# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
from eelbrain import *
from matplotlib import pyplot
import mne
import numpy
import re


# %run experiment.py           #e is the experiment object

# %%
word_young = e.load_evoked_stc('young', epoch='word', model='word % inflected'
       , src='vol-7', inv='vec-3-MNE-0', parc='aparc+aseg')

# %%
#plot.GlassBrain.butterfly(word_young['srcm'].mean('case'), cmap='Reds')

# %%
word_old = e.load_evoked_stc('old', epoch='word', model='word % inflected'
       , src='vol-7', inv='vec-3-MNE-0', parc='aparc+aseg')

# %%
word_aphasia = e.load_evoked_stc('aphasia', epoch='word', model='word % inflected'
       , src='vol-7', inv='vec-3-MNE-0', parc='aparc+aseg')

# %%
ds= combine(word_aphasia, word_young)
             

# %%
t_stat_result = testnd.TTestIndependent(
    'eeg', 'group', 'aphasia', 'young', match=None, data=ds,
    pmin=0.05,  
    tstart=0.100,  
    tstop=0.600   
)


# %%
