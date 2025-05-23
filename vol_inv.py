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
import eelbrain 
from matplotlib import pyplot
import mne
import numpy
import re
import os
from eelbrain import save, testnd


# %run experiment.py           #e is the experiment object

# %%
epochs=e.load_epochs(epoch='word-real') #for one subject by default
#epochs
tstart=0.1
tstop=0.6


# %%
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
res_cache_dir = os.path.expanduser("~/Code/MEGAN/Tests/oneSample")

session='words'
group='young'
epochs= ['word-real','word-stem','word-inflected']
#epoch word-real contains both realinflected and realuninflected

conditions=['pseudouninflected', 'realuninflected', 'pseudoinflected', 'realinflected']

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
    
   
    output_path = os.path.join(res_cache_dir, f"{cond}_{res_cache_file}.pickle")

    if os.path.exists(output_path):
        print(f"Loading cached result for: {cond}")
        res = eelbrain.load.unpickle(output_path)
    else:
        print(f"Running test for: {cond}")
        data = stc_all.sub(f"(lexical == '{cond}')")
        res = testnd.Vector('srcm', match='subject', data=data, tfce=True, tstart=tstart, tstop=tstop)
        save.pickle(res, output_path)

    print(f"{session}: {cond}\n{res.find_clusters()}")
    
       

# %% [markdown]
# # Plot Signigficant Clusters

# %%
#real-inf
clus=res.find_clusters(0.05, maps=True)
clus

# %%
#Plot first cluster of pseudo-inflected
#0	1	532	ctx-lh-inferiortemporal	0.1	0.6	0.5	0.0001	***
#plot.GlassBrain.butterfly(clus[0,'cluster'])


# %%
#plot.GlassBrain.butterfly(res)
#interactive (run from terminal)

# %%
#Diff real-inflected
plot.GlassBrain(res.masked_difference().sub(time=0.55),title=f"{cond}, 550 ms")

# %% [markdown]
# # Paired Tests
#

# %%
results_dir = os.path.expanduser("~/Code/MEGAN/Tests/PairedTests")
os.makedirs(results_dir, exist_ok=True)  

# %%
#e.load_events()
#stc_all.head()
contrast = ['real-pseudo','realinflected-realuninflected','pseudoinflected-pseudouninflected']
groups = ['young','old','aphasia','controls'] #Control=Young+old


# %% [markdown]
# ### RUN TESTS OR RELOAD

# %%
for group in groups:
    
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

    
    for pair in contrast:
        
        cond1, cond2 = pair.split('-')
        filename = f"{group}_{pair}.pickle"
        output_path = os.path.join(results_dir, filename)

        if pair == 'real-pseudo':
            col = 'word'
        else:
            col = 'lexical'
        if os.path.exists(output_path):
            print(f" Loading cached paired test result: {filename}")
            res = load.unpickle(output_path)
        else:
            print(f" Running VectorDifferenceRelated: {cond1} vs {cond2}")
            res = testnd.VectorDifferenceRelated(
                'srcm', col, cond1, cond2,
                match='subject', data=stc_all, tfce=True, tstart=tstart, tstop=tstop
                )
            save.pickle(res, output_path)

        print(f"{group}: {cond1} vs {cond2}\n{res.find_clusters()}")

#for table.difference we can't do it inline with testnd

#res = testnd.VectorDifferenceRelated('srcm', 'word', cond1, cond2, match='subject', data=stc_all, tfce=True, tstart=tstart, tstop=tstop)

# %% [markdown]
# ### PLOT RESULTS

# %% [markdown]
# #### Young, REAL VS Pseudo

# %%
pair='real-pseudo'
filename = f"{group}_{pair}.pickle"
output_path = os.path.join(results_dir, filename)
res = load.unpickle(output_path)


diff = res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k')
times = [0.45, 0.48, 0.52, 0.56]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(diff.sub(time=t), title=f"{group},{contrast}, {t*1000:.0f} ms")    

# %% [markdown]
# #### real-inf Vs. real-uninf
#

# %%
pair = 'realinflected-realuninflected'
filename = f"{group}_{pair}.pickle"
output_path = os.path.join(results_dir, filename)
res = load.unpickle(output_path)

# %%
diff = res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k')
times = [0.76]
for t in times:
    p.add_vline(t)
    f = plot.GlassBrain(diff.sub(time=t), title=f"{group}{contrast}, {t*1000:.0f} ms")  

# %% [markdown]
# #### psuedo-inf Vs. pseudo-uninf

# %%
pair = 'pseudoinflected-pseudouninflected'
filename = f"{group}_{pair}.pickle"
output_path = os.path.join(results_dir, filename)
res = load.unpickle(output_path)

# %%
diff = res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k')
times = [ 0.29,0.36, 0.42,0.58]
for t in times:
    p.add_vline(t)
    f = plot.GlassBrain(diff.sub(time=t), title=f"Pseudo-inflected Vs. Pseudo-uninflected ,{t*1000:.0f} ms")

# %%
#res = testnd.VectorDifferenceRelated('srcm', 'word % inflected', ('real', 'inflected'), ('pseudo', 'uninflected'), match='subject', data=stc_all, tfce=True, tstart=tstart, tstop=tstop)
#res = testnd.VectorDifferenceRelated('srcm', 'word', 'real', 'pseudo', sub="inflected == 'uninflected'", match='subject', data=stc_all, tfce=True, tstart=tstart, tstop=tstop)


# %% [markdown]
# ## Inside Aphasia Group


# %%
group='aphasia'



# %% [markdown]
# ### Real Vs. Pseudo

# %%
pair='real-pseudo'
filename = f"{group}_{pair}.pickle"
output_path = os.path.join(results_dir, filename)
res = load.unpickle(output_path)

# %%
diff= res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k')
#times = [0.29, 0.36, 0.42, 0.58]
#for t in times:
#    p.add_vline(t)
#for t in times:
#    f = plot.GlassBrain(diff.sub(time=t), title=f"Aphasia group, {contrast}, {t*1000:.0f} ms")

# %% [markdown]
# ### pseudo-inf VS pseudo-uninf

# %%
pair = 'pseudoinflected-pseudouninflected'
filename = f"{group}_{pair}.pickle"
output_path = os.path.join(results_dir, filename)
res = load.unpickle(output_path)

# %%
diff = res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k', title=f"Aphasia,{contrast}")
#times = [ 0.29,0.36, 0.42,0.58]
#for t in times:
#    p.add_vline(t)
#    f = plot.GlassBrain(diff.sub(time=t), title=f"Aphasia,{contrast} ,{t*1000:.0f} ms")

# %% [markdown]
# ### real-inf Vs. Real-uninf

# %%
pair = 'realinflected-realuninflected'
filename = f"{group}_{pair}.pickle"
output_path = os.path.join(results_dir, filename)
res = load.unpickle(output_path)

# %%
diff = res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k', title=f"Aphasia,{contrast}")
#times = [ 0.29,0.36, 0.42,0.58]

# %% [markdown]
# ## Inside Old Group

# %%
group='old'

# %%
#plot.GlassBrain.butterfly(old_stc_all['srcm'].mean('case'))

# %% [markdown]
# ### Real Vs. Pseudo

# %%
pair='real-pseudo'
filename = f"{group}_{pair}.pickle"
output_path = os.path.join(results_dir, filename)
res = load.unpickle(output_path)


# %%
#plot diff
diff = res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k')
#times = [0.29, 0.36, 0.42, 0.58]
#for t in times:
#    p.add_vline(t)
#for t in times:
#    f = plot.GlassBrain(diff.sub(time=t), title=f"{group} group, {contrast}, {t*1000:.0f} ms")

# %% [markdown]
# ### Pseudo : inf Vs Uninf

# %%
pair='pseudoinflected-pseudouninflected'
filename = f"{group}_{pair}.pickle"
output_path = os.path.join(results_dir, filename)
res = load.unpickle(output_path)

# %%
diff = res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k', title=f"{group},{contrast}")
#times = [ 0.29,0.36, 0.42,0.58]
#for t in times:
#    p.add_vline(t)
#    f = plot.GlassBrain(diff.sub(time=t), title=f{group},{contrast} ,{t*1000:.0f} ms")

# %% [markdown]
# ### Real: inf Vs Uninf

# %%
pair='realinflected-realuninflected'
filename = f"{group}_{pair}.pickle"
output_path = os.path.join(results_dir, filename)
res = load.unpickle(output_path)

# %%
#plot diff
diff = res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k',title=f"{group},{contrast}")
times = [0.29, 0.36, 0.42, 0.58]
#for t in times:
#    p.add_vline(t)
#for t in times:
#    f = plot.GlassBrain(diff.sub(time=t), title=f"{group} group, {contrast}, {t*1000:.0f} ms")

# %% [markdown]
# ## Inside Control Group : Young+OLD

# %%
group='controls'
pair='real-pseudo'
filename = f"{group}_{pair}.pickle"
output_path = os.path.join(results_dir, filename)
res = load.unpickle(output_path)

# %%
diff = res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k')
times = [ 0.27,0.29,0.38, 0.56,0.58]
for t in times:
    p.add_vline(t)
    f = plot.GlassBrain(diff.sub(time=t), title=f"{group},{pair},{t*1000:.0f} ms")

# %% [markdown]
# #### Controls: real-inf vs real uninf

# %%
pair='realinflected-realuninflected'
filename = f"{group}_{pair}.pickle"
output_path = os.path.join(results_dir, filename)
res = load.unpickle(output_path)

# %%
diff = res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k')
times = [ 0.49,0.52,0.58]
for t in times:
    p.add_vline(t)
    f = plot.GlassBrain(diff.sub(time=t), title=f"{group},{pair},{t*1000:.0f} ms")

# %% [markdown]
# #### Control, pseudo-inf vs pseudo uninf

# %%
pair='pseudoinflected-pseudouninflected'
filename = f"{group}_{pair}.pickle"
output_path = os.path.join(results_dir, filename)
res = load.unpickle(output_path)

# %%
diff = res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k')
times = [ 0.28,0.37,0.41,0.48,0.52,0.58]
for t in times:
    p.add_vline(t)
    f = plot.GlassBrain(diff.sub(time=t), title=f"{group},{pair},{t*1000:.0f} ms")

# %%
#TODO
#Find significant ROI then compare between groups 
