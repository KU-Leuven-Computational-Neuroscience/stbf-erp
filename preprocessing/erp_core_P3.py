import os
import mne

study_name = 'STBF-ERP_CORE'
bids_root = os.path.join(mne.get_config('MNE_DATA'), 'ERP_CORE_BIDS_Raw_Files')
deriv_root = os.path.join(mne.get_config('MNE_DATA'), 'derivatives', 'mne-bids-pipeline', 'STBF-ERP_CORE')

task = 'P3'
sessions = [task]

subjects = ['001', '002', '003']
#subjects = ['{:03d}'.format(s+1) for s in range(40)]

ch_types = ['eeg']
interactive = False

resample_sfreq = 64

eeg_template_montage = mne.channels.make_standard_montage('standard_1005')
eeg_bipolar_channels = {'HEOG': ('HEOG_left', 'HEOG_right'),
                        'VEOG': ('VEOG_lower', 'FP2')}
drop_channels = ['HEOG_left', 'HEOG_right', 'VEOG_lower']
eog_channels = ['HEOG', 'VEOG']

l_freq = 0.5
h_freq = 16

decode = False

find_breaks = True
min_break_duration = 10
t_break_annot_start_after_previous_event = 3.0
t_break_annot_stop_before_next_event = 1.5

ica_reject = dict(eeg=350e-6, eog=500e-6)
#reject = 'autoreject_global'
reject=None

#spatial_filter = 'ica'
spatial_filter=None
ica_max_iterations = 1000
ica_eog_threshold = 2

run_source_estimation = False

on_error = 'abort'
on_rename_missing_events = 'warn'
N_JOBS = 8

rename_events = {
    'response/201': 'response/correct',
    'response/202': 'response/incorrect',

    'stimulus/11': 'stimulus/target/11',
    'stimulus/22': 'stimulus/target/22',
    'stimulus/33': 'stimulus/target/33',
    'stimulus/44': 'stimulus/target/44',
    'stimulus/55': 'stimulus/target/55',
    'stimulus/21': 'stimulus/non-target/21',
    'stimulus/31': 'stimulus/non-target/31',
    'stimulus/41': 'stimulus/non-target/41',
    'stimulus/51': 'stimulus/non-target/51',
    'stimulus/12': 'stimulus/non-target/12',
    'stimulus/32': 'stimulus/non-target/32',
    'stimulus/42': 'stimulus/non-target/42',
    'stimulus/52': 'stimulus/non-target/52',
    'stimulus/13': 'stimulus/non-target/13',
    'stimulus/23': 'stimulus/non-target/23',
    'stimulus/43': 'stimulus/non-target/43',
    'stimulus/53': 'stimulus/non-target/53',
    'stimulus/14': 'stimulus/non-target/14',
    'stimulus/24': 'stimulus/non-target/24',
    'stimulus/34': 'stimulus/non-target/34',
    'stimulus/54': 'stimulus/non-target/54',
    'stimulus/15': 'stimulus/non-target/15',
    'stimulus/25': 'stimulus/non-target/25',
    'stimulus/35': 'stimulus/non-target/35',
    'stimulus/45': 'stimulus/non-target/45'
}
event_repeated = 'drop'
eeg_reference = ['P9', 'P10']
ica_n_components = 30 - len(eeg_reference)
epochs_tmin = -0.2
epochs_tmax = 1.0
baseline = None
conditions = ['stimulus/target', 'stimulus/non-target']
contrasts = [('stimulus/target', 'stimulus/non-target')]
