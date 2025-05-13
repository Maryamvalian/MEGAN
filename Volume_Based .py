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
epochs=e.load_epochs(epoch='word-real') #for one subject by default
#epochs
tstart=0.1
tstop=0.6

# %% jupyter={"source_hidden": true}
#word_young = e.load_evoked_stc('young', epoch='word', model='word % inflected'
#       , src='vol-5', inv='vec-3-MNE-0', parc='aparc+aseg')
#word_aphasia = e.load_evoked_stc('aphasia', epoch='word', model='word % inflected'
#       , src='vol-5', inv='vec-3-MNE-0', parc='aparc+aseg')
#word_old = e.load_evoked_stc('old', epoch='word', model='word % inflected'
#       , src='vol-5', inv='vec-3-MNE-0', parc='aparc+aseg')
#plot.GlassBrain.butterfly(word_young['srcm'].mean('case'), cmap='Reds')

# %% [markdown]
# ## Mask define

# %% jupyter={"source_hidden": true}
wholebrain_cortical_subcortical_mask = [
 'Left-Cerebral-White-Matter',
 'Left-Lateral-Ventricle',
 'Left-Inf-Lat-Vent',
 'Left-Putamen',
 'Left-Hippocampus',
 'Left-Amygdala',
 'Left-Accumbens-area',
 'Left-vessel',
 'Right-Cerebral-White-Matter',
 'Right-Inf-Lat-Vent',
 'Right-Putamen',
 'Right-Hippocampus',
 'Right-Amygdala',
 'ctx-lh-bankssts',
 'ctx-lh-caudalanteriorcingulate',
 'ctx-lh-caudalmiddlefrontal',
 'ctx-lh-cuneus',
 'ctx-lh-entorhinal',
 'ctx-lh-fusiform',
 'ctx-lh-inferiorparietal',
 'ctx-lh-inferiortemporal',
 'ctx-lh-isthmuscingulate',
 'ctx-lh-lateraloccipital',
 'ctx-lh-lateralorbitofrontal',
 'ctx-lh-lingual',
 'ctx-lh-medialorbitofrontal',
 'ctx-lh-middletemporal',
 'ctx-lh-parahippocampal',
 'ctx-lh-paracentral',
 'ctx-lh-parsopercularis',
 'ctx-lh-parsorbitalis',
 'ctx-lh-parstriangularis',
 'ctx-lh-pericalcarine',
 'ctx-lh-postcentral',
 'ctx-lh-posteriorcingulate',
 'ctx-lh-precentral',
 'ctx-lh-precuneus',
 'ctx-lh-rostralanteriorcingulate',
 'ctx-lh-rostralmiddlefrontal',
 'ctx-lh-superiorfrontal',
 'ctx-lh-superiorparietal',
 'ctx-lh-superiortemporal',
 'ctx-lh-supramarginal',
 'ctx-lh-frontalpole',
 'ctx-lh-temporalpole',
 'ctx-lh-transversetemporal',
 'ctx-lh-insula',
 'ctx-rh-bankssts',
 'ctx-rh-caudalanteriorcingulate',
 'ctx-rh-caudalmiddlefrontal',
 'ctx-rh-cuneus',
 'ctx-rh-entorhinal',
 'ctx-rh-fusiform',
 'ctx-rh-inferiorparietal',
 'ctx-rh-inferiortemporal',
 'ctx-rh-isthmuscingulate',
 'ctx-rh-lateraloccipital',
 'ctx-rh-lateralorbitofrontal',
 'ctx-rh-lingual',
 'ctx-rh-medialorbitofrontal',
 'ctx-rh-middletemporal',
 'ctx-rh-parahippocampal',
 'ctx-rh-paracentral',
 'ctx-rh-parsopercularis',
 'ctx-rh-parsorbitalis',
 'ctx-rh-parstriangularis',
 'ctx-rh-pericalcarine',
 'ctx-rh-postcentral',
 'ctx-rh-posteriorcingulate',
 'ctx-rh-precentral',
 'ctx-rh-precuneus',
 'ctx-rh-rostralanteriorcingulate',
 'ctx-rh-rostralmiddlefrontal',
 'ctx-rh-superiorfrontal',
 'ctx-rh-superiorparietal',
 'ctx-rh-superiortemporal',
 'ctx-rh-supramarginal',
 'ctx-rh-frontalpole',
 'ctx-rh-temporalpole',
 'ctx-rh-transversetemporal',
 'ctx-rh-insula']

lh_cortical_subcortical_mask = [
 'Left-Cerebral-White-Matter',
 'Left-Lateral-Ventricle',
 'Left-Inf-Lat-Vent',
 'Left-Putamen',
 'Left-Hippocampus',
 'Left-Amygdala',
 'Left-Accumbens-area',
 'Left-vessel',
 'ctx-lh-bankssts',
 'ctx-lh-caudalanteriorcingulate',
 'ctx-lh-caudalmiddlefrontal',
 'ctx-lh-cuneus',
 'ctx-lh-entorhinal',
 'ctx-lh-fusiform',
 'ctx-lh-inferiorparietal',
 'ctx-lh-inferiortemporal',
 'ctx-lh-isthmuscingulate',
 'ctx-lh-lateraloccipital',
 'ctx-lh-lateralorbitofrontal',
 'ctx-lh-lingual',
 'ctx-lh-medialorbitofrontal',
 'ctx-lh-middletemporal',
 'ctx-lh-parahippocampal',
 'ctx-lh-paracentral',
 'ctx-lh-parsopercularis',
 'ctx-lh-parsorbitalis',
 'ctx-lh-parstriangularis',
 'ctx-lh-pericalcarine',
 'ctx-lh-postcentral',
 'ctx-lh-posteriorcingulate',
 'ctx-lh-precentral',
 'ctx-lh-precuneus',
 'ctx-lh-rostralanteriorcingulate',
 'ctx-lh-rostralmiddlefrontal',
 'ctx-lh-superiorfrontal',
 'ctx-lh-superiorparietal',
 'ctx-lh-superiortemporal',
 'ctx-lh-supramarginal',
 'ctx-lh-frontalpole',
 'ctx-lh-temporalpole',
 'ctx-lh-transversetemporal',
 'ctx-lh-insula',
]


mask = wholebrain_cortical_subcortical_mask
mask_name = [name for name in globals() if globals()[name] is mask][0]

# %% [markdown]
# # One sample t-tests
# ## Inside Young Group


# %%
res_cache_file = f'  tfce young'
res_cache_dir ="~/Code/MEGAN/Tests/oneSample"

session='words'
group='young'
epochs= ['word-real','word-stem','word-inflected']
#epoch word-real contains both realinflected and realuninflected

conditions=['realinflected', 'realuninflected', 'pseudoinflected', 'pseudouninflected']

# From Lexical column 
# resolution 5 does not generate mask error	(low resolution like vol-10 generates error for mask detection										
e.set(epoch='word')
stc_all = e.load_evoked_stc(
    subjects=group, 
    baseline=False, 
    cov='emptyroom', 
    model='word % inflected',
    src='vol-10',
    inv='vec-3-MNE-0',
    parc='aparc+aseg',

)
#data['srcm']
#plot.Butterfly(data['srcm'].norm('space'))

for cond in conditions:
    
   
    data = stc_all.sub(f"(lexical == '{cond}')")
    res = testnd.Vector('srcm', match='subject', data=data, tfce=True, tstart=tstart, tstop=tstop)
    
    
    
    save.pickle(res, res_cache_dir + cond + res_cache_file)
    print(session + ': ' + cond + '\n' + str(res.find_clusters()))
    
       

# %% [markdown]
# # Plot Signigficant Clusters

# %%
clus=res.find_clusters(0.05, maps=True)

clus

# %%
#p = plot.TopoButterfly(res, t=0.13, head_radius=0.35)
#p = plot.TopoButterfly(res.tfce_map, t=0.13, head_radius=0.35)
res.masked_difference()
#p = plot.TopoButterfly(clus[0], t=0.30, head_radius=0.35).norm('space')
#plot.GlassBrain.butterfly(clus[0,'cluster'], cmap='Reds')
#plot.GlassBrain.butterfly(res, cmap='Reds')
plot.GlassBrain(res.masked_difference().sub(time=0.13), cmap='Reds')

# %% [markdown]
# # Paired Tests
# ## Inside 

# %%
e.load_events()

# %%
stc_all.head()

# %% [markdown]
# real vs pseudo
# real: inf vs real un inf
# psuedo: inf vs un-inf
#

# %%
contrast = ['realinflected-pseudouninflected']
cond1, cond2 = contrast[0].split('-')

#real_infl_stc = stc_all.sub(f"(lexical == '{cond1}')")
#pseudo_infl_stc = stc_all.sub(f"(lexical == '{cond2}')")
# both=stc_all.sub(f"(lexical == '{cond2}')|(lexical == '{cond1}')")
"""
diff_real_inf = table.difference(
        'srcm', 
        'subject',
        data=stc_all.sub(f"wordType=='{cond2}'"))
diff_pseudo_inf = table.difference(
        'srcm', 
        'subject',
        data=stc_all.sub(f"wordType=='{cond1}'"))

data_combined=combine([diff_real_inf, diff_pseudo_inf], incomplete='drop')
"""


res = testnd.VectorDifferenceRelated('srcm', 'lexical', cond1, cond2, match='subject', data=stc_all, tfce=True, tstart=tstart, tstop=tstop)

# %%
diff = res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k')
times = [0.13, 0.26, 0.35, 0.55]
for t in times:
    p.add_vline(t)

# %%
for t in times:
    p = plot.GlassBrain(diff.sub(time=t), cmap='Reds', title=f"{t*1000:.0f} ms")

# %%

res = testnd.VectorDifferenceRelated('srcm', 'word', 'real', 'pseudo', sub="inflected == 'uninflected'", match='subject', data=stc_all, tfce=True, tstart=tstart, tstop=tstop)


# %%

res = testnd.VectorDifferenceRelated('srcm', 'word % inflected', ('real', 'inflected'), ('pseudo', 'uninflected'), match='subject', data=stc_all, tfce=True, tstart=tstart, tstop=tstop)

