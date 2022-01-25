# Spatiotemporal beamforming

ERP classification using spatiotemporal beamforming as implemented in [1].
This software makes use of the [MNE-Python toolbox](https://mne.tools/stable/index.html) [2],
the [BIDS-EEG format](https://bids-specification.readthedocs.io/en/stable/) [3],
[MNE-BIDS](https://mne.tools/mne-bids/stable/index.html#) [4]
and [MNE-BIDS-Pipeline](https://mne.tools/mne-bids-pipeline/).
A comparative classifier [5] is implemented with [pyRiemann](https://pyriemann.readthedocs.io/en/latest/index.html)

## Installation
First, clone this git repository and change to its directory.
```shell
virtualenv -p python3 .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Setup
Make sure the `MNE_DATA` directory is correctly configured 
(https://mne.tools/stable/auto_tutorials/intro/50_configure_mne.html?highlight=configuration).
Download `ERP_CORE_BIDS_Raw_Files` from https://osf.io/thsqg/ and extract it to `$MNE_DATA/ERP_CORE_BIDS_Raw_Files`

Preprocess the ERP-CORE P3 dataset by executing
```shell
preprocessing/mne_bids_pipeline/run.py --config preprocessing/erp_core_P3.py
```

## References

[1] Van Den Kerchove, A.; Libert, A.; Wittevrongel, B.; Van Hulle, M.M.
Classification of Event-Related Potentials with Regularized Spatiotemporal LCMV Beamforming.
Appl. Sci. 2022,

[2] Gramfort, A.; Luessi, M.; Larson, E.; Engemann, D.A.; Strohmeier, D.; Brodbeck, C.; Goj, R.; Jas, M.; Brooks, T.; Parkkonen, L.; 553
et al. MEG and EEG Data Analysis with MNE-Python. Frontiers in Neuroscience 2013, 7, 1–13. doi:10.3389/fnins.2013.00267.

[3] Pernet, C.R.; Appelhoff, S.; Gorgolewski, K.J.; Flandin, G.; Phillips, C.; Delorme, A.; Oostenveld, R. EEG-BIDS, an extension to the 555
brain imaging data structure for electroencephalography. Scientific Data 2019, 6, 103. doi:10.1038/s41597-019-0104-8.

[4] Appelhoff, S.; Sanderson, M.; Brooks, T.L.; Vliet, M.v.; Quentin, R.; Holdgraf, C.; Chaumon, M.; Mikulan, E.; Tavabi, K.; 557
Höchenberger, R.; et al. MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. 558
Journal of Open Source Software 2019, 4, 1896. doi:10.21105/joss.01896.

[5] Barachant, A. MEG decoding using Riemannian Geometry and Unsupervised classification. Grenoble University: Grenoble, France 560 2014.