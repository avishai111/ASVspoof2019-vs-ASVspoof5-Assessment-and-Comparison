from GammatoneFilter import GammatoneFilterbank
from PMF import PMF
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.io import loadmat
import json
import PMF_measure_utils 
from checks.data_loading import load_data_male, get_real_channel,get_columns_names_feature_importance
import pandas as pd

# checks, Still incomplete.


NUM_BINS = 2**16
HIST_EDGES = (-0.999969482421875, 1) # Adjusted from Matan's Code: it's for the 0-bin to exist
TRAIN_FILE_FOLDER = 'C:/Users/avish/OneDrive/Desktop/avishai111-ASVspoof2019-vs-ASVspoof5-Assessment-and-Comparison/PMF_based_embeddings_implemention/LA/ASVspoof2019_LA_train/flac/'  # Replace with your actual folder path

PROTOCOL_TRAIN ='C:/Users/avish/OneDrive/Desktop/avishai111-ASVspoof2019-vs-ASVspoof5-Assessment-and-Comparison/PMF_based_embeddings_implemention/ASVspoof2019/ASVspoof2019_test.LA.cm.train.trn.txt'

DEV_FILE_FOLDER = "/Users/guyperets/Documents/MSc/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_dev/flac"
PROTOCOL_DEV = "/Users/guyperets/Documents/MSc/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"

EVAL_FILE_FOLDER = "/Users/guyperets/Documents/MSc/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_eval/flac"
PROTOCOL_EVAL = "/Users/guyperets/Documents/MSc/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

MATAN_HIST_DETAILS = "/Users/guyperets/Documents/MSc/PythonNotebooks/MatanPythonConversion/pmf_after_filters/ASVSpoof2019/gammtone/hist_details.mat"
MATAN_RESULT_JSON = "/Users/guyperets/Documents/MSc/PythonNotebooks/MatanPythonConversion/GuyPythonConversion/hist_data_formatted.json" # Created from what Avishai sent me, which is a conversion of Matan's hist_details.mat file
MATLAB_GAMMATONE_FILTERS = "/Users/guyperets/Documents/MSc/ASVspoof_PMF-2d_quantization/matlab/filters/gammatone_filters.mat" # Generated from Matan's Code (Ask Guy-P for create_filters_guy.m file if needed)


if __name__ == "__main__":
    embedded_groups_1_1, _, _, \
    _, _, _, \
    _, _, _, \
    _, _, _, \
    chosen_labels_1_1_name, _, _, \
    _, _, _, \
    male_chosen_labels_1_1_sex, _, _ = load_data_male("./PMF_based_embeddings_implemention/checks/male/")
    
    columns_names,max_name_length = get_columns_names_feature_importance(substruct=True)
    true_channels_indexes = np.array(get_real_channel(np.linspace(start=1, stop=len(columns_names), num=len(columns_names)),len(columns_names)))
    true_channels_indexes = true_channels_indexes - 1
    true_channels_indexes = true_channels_indexes.astype(int)
    columns_names = np.array(columns_names)
    embedded_groups_1_1 = embedded_groups_1_1[:,true_channels_indexes]
    
    
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
    # pmf_train_spoof, edges_train_spoof = pmf_t.compute_hist_by_category_stream("spoof", num_bins=NUM_BINS, hist_edges=HIST_EDGES)
    
    # pmf_train_spoof = np.array([pmf for (_, pmf) in pmf_train_spoof])
    # pmf_train_bonafide, edges_train_bonafide = pmf_t.compute_hist_by_category_stream("bonafide", num_bins=NUM_BINS, hist_edges=HIST_EDGES)
    # pmf_train_bonafide = np.array([pmf for (_, pmf) in pmf_train_bonafide])
    data = np.load("pmf__histograms_train_data.npz")

    pmf_train_spoof = data["pmf_train_spoof"]
    edges_train_spoof = data["edges_train_spoof"]
    pmf_train_bonafide = data["pmf_train_bonafide"]
    edges_train_bonafide = data["edges_train_bonafide"]

    print("PMF histograms for training data computed.")
    # Compute PMF histograms for development speech
    file_path = os.path.join(TRAIN_FILE_FOLDER,"LA_T_1199930.flac")  # Example file, replace with actual file path
    res, filenames = pmf_t.compute_hist_per_input_file_stream(file_path,num_bins = NUM_BINS, hist_edges = HIST_EDGES)
    print("PMF histograms for training data computed.")
    res = res[None, :]  # Add a new axis to match the expected shape
    
    #calculate distnaces bettween the PMF histograms
    print("Calculating distances between PMF histograms...")
    dist_bona = PMF_measure_utils.compute_distances_to_reference(res, pmf_train_bonafide)
    dist_spoof = PMF_measure_utils.compute_distances_to_reference(res, pmf_train_spoof)

    # Compute difference metric-by-metric
    diff = {
        key: dist_spoof[key] - dist_bona[key]  for key in dist_bona
    }
    interleaved = []
    for i in range(20):  # 20 steps × 8 metrics = 160
        for key in diff.keys():
            value = diff[key].flatten()  # ensure it's a 1D array
            interleaved.append(value[i])

    diff_df = pd.DataFrame([interleaved], columns=columns_names)
    
    diff_df - embedded_groups_1_1[0,:]
    
    diff_array = diff_df.values.flatten()  # shape (160,)
    embedded_array = embedded_groups_1_1[0, :]  
  
    assert diff_array.shape == embedded_array.shape
    difference = diff_array - embedded_array
    print(f"Mean difference: {np.mean(difference):.6f}")
    print(f"Max difference: {np.max(np.abs(difference)):.6f}")
    print(diff_array)
    print(embedded_array)
    
