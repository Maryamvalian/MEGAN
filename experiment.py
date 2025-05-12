
"""
Empty room
----------
Code for generating empty room data:

    from pathlib import Path

    root = Path('~/Data/root/meg')
    src = root / 'ERT_10.16.12_9h25m-raw.fif'
    for sdir in root.glob('R????'):
        dst = sdir / f'{sdir.name}_emptyroom-raw.fif'
        src.link_to(dst)


Visits
------
Visit 2: pre-treatment to establish stability (2 patients missing)
Visit 3: post-treatment


Preprocessing
-------------
After taking over I additionally auto-rejected channels based on low neighbor-correlation

auto-reject epochs with absolute thershold 2e-12
- confirmed done for young

    e.set(raw='n-ica-14', epoch='word', group='good')
    for s in e:
        e.make_epoch_selection(auto=2e-12)
    e.set(visit='3', group='aphasia')
    for s in e:
        e.make_epoch_selection(auto=2e-12)


"""
from itertools import chain
from pathlib import Path
import warnings
# -

from eelbrain import *
from eelbrain.pipeline import *

# hide unimportant
warnings.filterwarnings('ignore', category=FutureWarning)


# Default since mne 0.16
FILTER_KWARGS = {
    'filter_length': 'auto',
    'l_trans_bandwidth': 'auto',
    'h_trans_bandwidth': 'auto',
    'phase': 'zero',
    'fir_window': 'hamming',
    'fir_design': 'firwin',
}

YOUNG = ('R0240', 'R0284', 'R0645', 'R0654', 'R0656', 'R0727', 'R0728', 'R0729', 'R0730', 'R0734', 'R0773', 'R0793', 'R0885', 'R0918')
limited_YOUNG = ('R0240', 'R0284','R0645','R0654')
OLD = ('R0735', 'R0742', 'R0744', 'R0783', 'R0784', 'R0786', 'R0840', 'R0850', 'R0886')
APHASIA = ('R0655', 'R0721', 'R0782', 'R0952', 'R0944', 'R1015')

IFG = ('parsorbitalis', 'parstriangularis', 'parsopercularis')
TEMPORAL = ('transversetemporal', 'superiortemporal', 'bankssts', 'middletemporal', 'inferiortemporal', 'temporalpole', 'fusiform', 'parahippocampal', 'entorhinal')
OCCIPITAL = ('cuneus', 'pericalcarine', 'lingual', 'lateraloccipital')

#ROOT = Path('dataset').expanduser()
ROOT = Path('~/Data/Aphasia').expanduser()


class Experiment(MneExperiment):

    path_version = 1
    auto_delete_cache = 'debug'
    check_raw_mtime = False

    visits = ('', '2', '3')

    sessions = ('words', 'sentences', 'emptyroom')

    defaults = {
        'raw': 'n-ica',
        #Mary 
        'group': 'young'
    }

    raw = {
        'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=.9, st_only=True),
        # original pipeline !! ICA without filtering
        '1-40': RawFilter('tsss', 1, 40, **FILTER_KWARGS, cache=False),
        'ica': RawICA('tsss', ('words', 'sentences'), n_components=0.99, random_state=0),
        'ica1-40': RawFilter('ica', 1, 40, **FILTER_KWARGS, cache=False),
        'ica1-20': RawFilter('ica', 1, 20, **FILTER_KWARGS, cache=False),
        # secondary ICA
        'n-1-20': RawFilter('tsss', 1, 20),
        'n-ica': RawICA('n-1-20', ('words', 'sentences'), random_state=0),
        'n-unfiltered': RawApplyICA('tsss', 'n-ica'),
        'n-ica-14': RawFilter('n-ica', 0, 14),
        'n-picard': RawICA('n-1-20', ('words', 'sentences'), 'picard', random_state=0, fit_params=dict(ortho=False, extended=True)),
        # OTP
        'filt-1-80': RawFilter('tsss', 1, 80, cache=False),
        '1-80': RawApplyICA('filt-1-80', 'n-ica'),
    }

    def fix_events(self, ds):
        if ds.info['session'] == 'emptyroom':
            return ds
        if ds[1, 'trigger'] <= 161:
            ds['trigger'] -= 161
            ds = ds.sub("trigger > 160")
        elif ds.info['subject'] == 'R0782':
            ds = ds.sub("trigger <= 165")
            ds = ds.sub("trigger > 161")
        return ds

    def label_events(self, ds):
        if ds.info['session'] == 'words':
            ds['word'] = Factor(ds['trigger'], labels={162: 'real', 163: 'pseudo', 164: 'pseudo', 165: 'real'})
            ds['inflected'] = Factor(ds['trigger'], labels={162: 'inflected', 163: 'inflected', 164: 'uninflected', 165: 'uninflected'})
            ds['lexical'] = Factor(ds['trigger'], labels={162: 'realinflected', 163: 'pseudoinflected', 164: 'pseudouninflected', 165: 'realuninflected'})
        elif ds.info['session'] == 'sentences':
            ds['felicity'] = Factor(ds['trigger'], labels={162: 'bad', 163: 'good', 164: 'bad', 165: 'bad'})
            ds['violation'] = Factor(ds['trigger'], labels={162: 'tense', 163: 'none', 164: 'semantic', 165: 'grammatical'})
        elif ds.info['session'] != 'emptyroom':
            raise RuntimeError(f"unknown session {ds.info['session']}")
        return ds

    epochs = {
        'word': PrimaryEpoch('words', tmax=0.8, samplingrate=250),
        'word-real': SecondaryEpoch('word', "word == 'real'"),
        'word-stem': SecondaryEpoch('word', "inflected == 'uninflected'"),
        'word-inflected': SecondaryEpoch('word', "inflected == 'inflected'"),
        'cov': SecondaryEpoch('word', tmax=0),
        'sentence': PrimaryEpoch('sentences', tmax=1.0, samplingrate=250),
        'sentence-shared': SecondaryEpoch('sentence', "violation != 'grammatical'"),
        'sentence-none': SecondaryEpoch('sentence', "violation == 'none'"),
        'sentence-semantic': SecondaryEpoch('sentence', "violation == 'semantic'"),
    }

    groups = {
        'bad': ['R0637'],
        'good': SubGroup('all', ['R0637']),
        'young': YOUNG,
        'old': OLD,
        'controls': YOUNG + OLD,
        'aphasia': APHASIA,
        'controls165': ('R0727', 'R0728', 'R0729', 'R0730', 'R0734', 'R0735', 'R0742', 'R0744', 'R0773', 'R0783', 'R0784', 'R0786', 'R0793', 'R0840', 'R0850', 'R0885', 'R0886', 'R0918'),
        'controlsno165': ('R0240', 'R0284', 'R0645', 'R0654', 'R0656'),
        'visit2': ('R0655', 'R0782', 'R0952', 'R1015'),

        'small': limited_YOUNG,
    }
    
    variables = {
        'group': GroupVar(['young', 'old', 'aphasia', 'bad']),
    }

    parcs = {
        'fusiform-lh': SubParc('aparc', ['fusiform-lh']),
        'fusiform-rh': SubParc('aparc', ['fusiform-rh']),
        'ifg-lh': SubParc('aparc', [f'{label}-lh' for label in IFG]),
        'ifg-rh': SubParc('aparc', [f'{label}-rh' for label in IFG]),

        'ot': SubParc('aparc', list(chain(TEMPORAL, OCCIPITAL)), 'lateral'),
        'ot-lh': SubParc('aparc', [f'{label}-lh' for label in chain(TEMPORAL, OCCIPITAL)], 'lateral'),


        'frontotemporal-lh': CombinationParc('aparc', {'frontal-lh': 'superiorfrontal + rostralmiddlefrontal + caudalmiddlefrontal+ parsorbitalis + parstriangularis + parsopercularis + lateralorbitofrontal + medialorbitofrontal + precentral + frontalpole + paracentral', 'temporal-lh': 'transversetemporal + superiortemporal + middletemporal + inferiortemporal + bankssts', }, views='lateral'),
        'frontotemporal-rh': CombinationParc('aparc', {'frontal-rh': 'superiorfrontal + rostralmiddlefrontal + caudalmiddlefrontal+ parsorbitalis + parstriangularis + parsopercularis + lateralorbitofrontal + medialorbitofrontal + precentral + frontalpole + paracentral', 'temporal-rh': 'transversetemporal + superiortemporal + middletemporal + inferiortemporal + bankssts', }, views='lateral'),

        'temporal-rh': CombinationParc('aparc', {'temporal-rh': 'transversetemporal + superiortemporal + middletemporal + inferiortemporal + bankssts', }, views='lateral'),
        'temporal-lh': CombinationParc('aparc', {'temporal-lh': 'transversetemporal + superiortemporal + middletemporal + inferiortemporal + bankssts', }, views='lateral'),
        'temporal2-lh': CombinationParc('aparc', {'superiortemporal-lh': 'transversetemporal + superiortemporal + bankssts', 'middletemporal-lh': 'middletemporal', 'inferiortemporal-lh': 'inferiortemporal', }, views='lateral'),

        'frontal2-lh': CombinationParc('aparc', {'inferiorfrontal-lh': 'superiorfrontal + rostralmiddlefrontal', 'superiorfrontal-lh': 'caudalmiddlefrontal + parsorbitalis + parstriangularis'}),
        'frontal-rh': CombinationParc('aparc', {'frontal-rh': 'superiorfrontal+ rostralmiddlefrontal+ caudalmiddlefrontal+ parsorbitalis + parstriangularis + parsopercularis + lateralorbitofrontal + medialorbitofrontal + precentral + frontalpole + paracentral'}),
        'frontal-lh': CombinationParc('aparc', {'frontal-lh': 'superiorfrontal+ rostralmiddlefrontal+ caudalmiddlefrontal+ parsorbitalis + parstriangularis + parsopercularis + lateralorbitofrontal + medialorbitofrontal + precentral + frontalpole + paracentral'}),
        'inferiorfrontal-lh': CombinationParc('aparc', {'inferiorfrontal-lh': 'parsorbitalis + parstriangularis + parsopercularis'}),

        'middletemporal-lh': CombinationParc('aparc', {'middletemporal': 'middletemporal'}),
    }

    tests = {
        '0': TTestOneSample(),

        # lexical decision
        ###################
        'inflection': TTestRelated('inflected', 'inflected', 'uninflected'),
        'lexical': TTestRelated('word', 'real', 'pseudo'),

        # early visual
        'early': TTestRelated('inflected', 'inflected', 'uninflected', tail=1),

        # ANOVAS
        'pseudoxinflection': ANOVA('inflected * word * subject'),
        'lexicalxsubject': ANOVA('lexical * subject'),


        # sentence judgment
        ####################
        # Violations
        'correct': TTestRelated('grammaticality', 'correct', 'incorrect'),
        'tense': TTestRelated('violation', 'tense', 'none'),
        'semantic': TTestRelated('violation', 'semantic', 'none'),
        'tense=semantic': TTestRelated('violation', 'tense', 'semantic'),

        # ANOVA
        'violation': ANOVA('violation * subject'),


        # group differences
        ####################
        # group difference TBD
        'age': TTestIndependent('group', 'old', 'young'),
        'aphasia-controls': TTestIndependent('group', 'aphasia', 'controls'),
        'aphasia-old': TTestIndependent('group', 'aphasia', 'old'),
    }


e = Experiment(ROOT)












