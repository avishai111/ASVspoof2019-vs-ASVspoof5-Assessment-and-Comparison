from GammatoneFilter import GammatoneFilterbank
from PMF import PMF
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.io import loadmat
import json
import PMF_measure_utils 
NUM_BINS = 2**16
HIST_EDGES = (-0.999969482421875, 1) # Adjusted from Matan's Code: it's for the 0-bin to exist
TRAIN_FILE_FOLDER = 'C:/Users/avish/OneDrive/Desktop/avishai111-ASVspoof2019-vs-ASVspoof5-Assessment-and-Comparison/PMF_based_embeddings_implemention/LA/ASVspoof2019_LA_train/flac/'  # Replace with your actual folder path

PROTOCOL_TRAIN ='C:/Users/avish/OneDrive/Desktop/avishai111-ASVspoof2019-vs-ASVspoof5-Assessment-and-Comparison/PMF_based_embeddings_implemention/ASVspoof2019/ASVspoof2019.LA.cm.train.trn.txt'

DEV_FILE_FOLDER = "/Users/guyperets/Documents/MSc/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_dev/flac"
PROTOCOL_DEV = "/Users/guyperets/Documents/MSc/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"

EVAL_FILE_FOLDER = "/Users/guyperets/Documents/MSc/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_eval/flac"
PROTOCOL_EVAL = "/Users/guyperets/Documents/MSc/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

MATAN_HIST_DETAILS = "/Users/guyperets/Documents/MSc/PythonNotebooks/MatanPythonConversion/pmf_after_filters/ASVSpoof2019/gammtone/hist_details.mat"
MATAN_RESULT_JSON = "/Users/guyperets/Documents/MSc/PythonNotebooks/MatanPythonConversion/GuyPythonConversion/hist_data_formatted.json" # Created from what Avishai sent me, which is a conversion of Matan's hist_details.mat file
MATLAB_GAMMATONE_FILTERS = "/Users/guyperets/Documents/MSc/ASVspoof_PMF-2d_quantization/matlab/filters/gammatone_filters.mat" # Generated from Matan's Code (Ask Guy-P for create_filters_guy.m file if needed)


if __name__ == "__main__":
    SAVE_ref_flag = True
    LOAD_ref_flag = False
    SAVE_res_flag = True
    LOAD_res_flag = False
    res_type = 'train'
    
    #initialize the PMF object
    pmf_t , pmf_d, pmf_e = None, None, None
    pmf_train_spoof , pmf_train_bonafide = None, None
    edges_train_spoof, edges_train_bonafide = None, None
    res, filenames = None, None
    # Load the PMF data
    gfb = GammatoneFilterbank(
    num_filters=10,
    sample_rate=16e3,
    low_freq=0,
    high_freq=8e3,
    num_fft=2047, # From Matan's Code
    with_inverse=True
    )
    pmf_t = PMF(TRAIN_FILE_FOLDER, PROTOCOL_TRAIN, ftype=gfb)
    # pmf_d = PMF(DEV_FILE_FOLDER, PROTOCOL_DEV, ftype=gfb)
    # pmf_e = PMF(EVAL_FILE_FOLDER, PROTOCOL_EVAL, ftype=gfb)
    if res_type == 'train':
        res_PMF = pmf_t
    elif res_type == 'dev':
        res_PMF = pmf_d
    elif res_type == 'eval':
        res_PMF = pmf_e

    # caluclate the PMF histograms for training set
    print("Computing PMF histograms for training set...")
    '''
    [
    (hist_channel_0, pmf_channel_0),  # hist_channel_0: (65536,), pmf_channel_0: (65536,)
    (hist_channel_1, pmf_channel_1),
    ...
    (hist_channel_19, pmf_channel_19)
    ]
    '''
    
    if SAVE_ref_flag:
        pmf_train_spoof, edges_train_spoof = pmf_t.compute_hist_by_category_stream("spoof", num_bins=NUM_BINS, hist_edges=HIST_EDGES)
        pmf_train_bonafide, edges_train_bonafide = pmf_t.compute_hist_by_category_stream("bonafide", num_bins=NUM_BINS, hist_edges=HIST_EDGES)

        np.savez("pmf_train_data.npz",
                 pmf_train_spoof=pmf_train_spoof,
                 edges_train_spoof=edges_train_spoof,
                 pmf_train_bonafide=pmf_train_bonafide,
                 edges_train_bonafide=edges_train_bonafide)

    
    if LOAD_ref_flag:
        data = np.load("pmf_train_data.npz")
        pmf_train_spoof = data["pmf_train_spoof"]
        edges_train_spoof = data["edges_train_spoof"]
        pmf_train_bonafide = data["pmf_train_bonafide"]
        edges_train_bonafide = data["edges_train_bonafide"]
        
    print("PMF histograms for training data computed.")
    # Compute PMF histograms for development speech  
    if SAVE_res_flag:
        res, filenames = res_PMF.compute_hist_per_file_stream(num_bins=NUM_BINS, hist_edges=HIST_EDGES)
        if res_type == 'train':
            np.savez("pmf_res_train_data.npz",
                    res=res,
                    filenames=filenames)
        elif res_type == 'dev':
            np.savez("pmf_res_dev_data.npz",
                    res=res,
                    filenames=filenames)
        elif res_type == 'eval':
            np.savez("pmf_res_eval_data.npz",
                    res=res,
                    filenames=filenames)

    if LOAD_res_flag:
        if res_type == 'train':
            data = np.load("pmf_res_train_data.npz")
        elif res_type == 'dev':
            data = np.load("pmf_res_dev_data.npz")
        elif res_type == 'eval':
            data = np.load("pmf_res_eval_data.npz")
            
        res = data["res"]
        filenames = data["filenames"]

    print("PMF histograms for development data computed.")

    #calculate distnaces bettween the PMF histograms
    print("Calculating distances between PMF histograms...")
    dist_bona = PMF_measure_utils.compute_distances_to_reference(res, pmf_train_bonafide)
    dist_spoof = PMF_measure_utils.compute_distances_to_reference(res, pmf_train_spoof)

    # Compute difference metric-by-metric
    diff = {
        key: dist_spoof[key] - dist_bona[key]  for key in dist_bona
    }
    print("Calculating distances between PMF histograms...")
    