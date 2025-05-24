import sys
import os
import torch
import sklearn  
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

def read_ASVSpoof5_protocol(protocol_path):
    protocol = pd.read_csv(protocol_path, sep=' ', header=None)
    protocol.columns = ['SPEAKER_ID','FLAC_FILE_NAME', 'SPEAKER_GENDER', 'CODEC', 'CODEC_Q', 'CODEC_SEED', 'ATTACK_TAG', 'ATTACK_LABEL', 'KEY', 'TMP']
    protocol = protocol.drop(columns=['TMP'])
    return protocol

#loading the all data for loading the  PMF-based embeddings in ASVspoof05
def load_data_all_ASVSpoof5(data_path,include_eval=True,include_dev = True):
    chosen_labels_numeric_1_1,chosen_labels_numeric_2_1,chosen_labels_numeric_3_1 = pd.Series([]),pd.Series([]),pd.Series([])
    
    embedded_groups_1_1 = scipy.io.loadmat(os.path.join(data_path,'all_embedded_groups_1_1.mat'))['embedded_groups_1_1'];
    
    embedded_groups_1_2 = np.array([])
    if include_dev:
        embedded_groups_1_2 = scipy.io.loadmat(os.path.join(data_path,'all_embedded_groups_2_1.mat'))['embedded_groups_2_1'];
    
    embedded_groups_1_3 = np.array([])
    if include_eval:
        embedded_groups_1_3 = scipy.io.loadmat(os.path.join(data_path,'all_embedded_groups_3_1.mat'))['embedded_groups_3_1'];

    chosen_labels_1_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_is_spoofed.mat'))['chosen_labels_1_1_is_spoofed'];
    chosen_labels_1_1_is_spoofed = pd.Series([item for sublist in chosen_labels_1_1_is_spoofed for item in sublist]);

    chosen_labels_2_1_is_spoofed = pd.Series([])
    if include_dev:
        chosen_labels_2_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_is_spoofed.mat'))['chosen_labels_2_1_is_spoofed'];
        chosen_labels_2_1_is_spoofed = pd.Series([item for sublist in chosen_labels_2_1_is_spoofed for item in sublist]);

    chosen_labels_3_1_is_spoofed = pd.Series([])
    if include_eval:
        chosen_labels_3_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_is_spoofed_correct.mat'))['chosen_labels_3_1_is_spoofed'];
        chosen_labels_3_1_is_spoofed = pd.Series([item for sublist in chosen_labels_3_1_is_spoofed for item in sublist]);

    # chosen_labels_numeric_1_1 = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_numeric_1_1.mat'))['chosen_labels_numeric_1_1'][:];
    # chosen_labels_numeric_1_1 = pd.Series([item for sublist in chosen_labels_numeric_1_1 for item in sublist]);

    # chosen_labels_numeric_2_1 = pd.Series([])
    # if include_dev:
    #     chosen_labels_numeric_2_1 = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_numeric_2_1.mat'))['chosen_labels_numeric_2_1'][:];
    #     chosen_labels_numeric_2_1 = pd.Series([item for sublist in chosen_labels_numeric_2_1 for item in sublist]);
    
    # chosen_labels_numeric_3_1 = pd.Series([])
    # if include_eval:
    #     chosen_labels_numeric_3_1 = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_numeric_3_1.mat'))['chosen_labels_numeric_3_1'][:];
    #     chosen_labels_numeric_3_1 = pd.Series([item for sublist in chosen_labels_numeric_3_1 for item in sublist]);

    chosen_labels_1_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_attack_logical.mat'))['chosen_labels_1_1_attack_logical'];
    chosen_labels_1_1_attack_logical = pd.Series([item for sublist in chosen_labels_1_1_attack_logical for item in sublist])

    chosen_labels_2_1_attack_logical = pd.Series([])
    if include_dev:
        chosen_labels_2_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_attack_logical.mat'))['chosen_labels_2_1_attack_logical'];
        chosen_labels_2_1_attack_logical = pd.Series([item for sublist in chosen_labels_2_1_attack_logical for item in sublist])

    chosen_labels_3_1_attack_logical = pd.Series([])
    if include_eval:
        chosen_labels_3_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_attack_logical_correct.mat'))['chosen_labels_3_1_attack_logical'];
        chosen_labels_3_1_attack_logical = pd.Series([item for sublist in chosen_labels_3_1_attack_logical for item in sublist])

    chosen_labels_1_1_name = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_name.mat'))['chosen_labels_1_1_name'];
    chosen_labels_1_1_name = pd.Series([item[0] for sublist in chosen_labels_1_1_name for item in sublist])

    chosen_labels_2_1_name = pd.Series([])
    if include_dev:
        chosen_labels_2_1_name = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_name.mat'))['chosen_labels_2_1_name'];
        chosen_labels_2_1_name = pd.Series([item[0] for sublist in chosen_labels_2_1_name for item in sublist])
    
    chosen_labels_3_1_name = pd.Series([])
    if include_eval:
        chosen_labels_3_1_name = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_name_correct.mat'))['chosen_labels_3_1_name'];
        chosen_labels_3_1_name = pd.Series([item[0] for sublist in chosen_labels_3_1_name for item in sublist])

    chosen_labels_1_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_speaker_id.mat'))['chosen_labels_1_1_speaker_id'];
    chosen_labels_1_1_speaker_id = pd.Series([item for sublist in chosen_labels_1_1_speaker_id for item in sublist])

    chosen_labels_2_1_speaker_id = pd.Series([])
    if include_dev:
        chosen_labels_2_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_speaker_id.mat'))['chosen_labels_2_1_speaker_id'];
        chosen_labels_2_1_speaker_id = pd.Series([item for sublist in chosen_labels_2_1_speaker_id for item in sublist])
    
    chosen_labels_3_1_speaker_id = pd.Series([])
    if include_eval:
        chosen_labels_3_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_speaker_id_correct.mat'))['chosen_labels_3_1_speaker_id'];
        chosen_labels_3_1_speaker_id = pd.Series([item for sublist in chosen_labels_3_1_speaker_id for item in sublist])

    chosen_labels_1_1_sex = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_sex.mat'))['chosen_labels_1_1_sex'];
    chosen_labels_1_1_sex = pd.Series([item for sublist in chosen_labels_1_1_sex for item in sublist]);

    chosen_labels_2_1_sex = pd.Series([])
    if include_dev:
        chosen_labels_2_1_sex = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_sex.mat'))['chosen_labels_2_1_sex'];
        chosen_labels_2_1_sex = pd.Series([item for sublist in chosen_labels_2_1_sex for item in sublist]);
    
    chosen_labels_3_1_sex = pd.Series([])
    if include_eval:
        chosen_labels_3_1_sex = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_sex_correct.mat'))['chosen_labels_3_1_sex'];
        chosen_labels_3_1_sex = pd.Series([item for sublist in chosen_labels_3_1_sex for item in sublist]);
    
    
    return  embedded_groups_1_1,embedded_groups_1_2,embedded_groups_1_3, \
            chosen_labels_1_1_is_spoofed,chosen_labels_2_1_is_spoofed,chosen_labels_3_1_is_spoofed, \
            chosen_labels_numeric_1_1,chosen_labels_numeric_2_1,chosen_labels_numeric_3_1, \
            chosen_labels_1_1_attack_logical,chosen_labels_2_1_attack_logical,chosen_labels_3_1_attack_logical,chosen_labels_1_1_name, \
            chosen_labels_2_1_name,chosen_labels_3_1_name,chosen_labels_1_1_speaker_id,chosen_labels_2_1_speaker_id,chosen_labels_3_1_speaker_id, \
            chosen_labels_1_1_sex,chosen_labels_2_1_sex,chosen_labels_3_1_sex   
            
            

#loading the male data for loading the  PMF-based embeddings for male gender in ASVspoof2019
def load_data_male(data_path):
    embedded_groups_1_1 = scipy.io.loadmat(os.path.join(data_path,'male_embedded_groups_1_1.mat'))['embedded_groups_1_1'];
    embedded_groups_1_2 = scipy.io.loadmat(os.path.join(data_path,'male_embedded_groups_2_1.mat'))['embedded_groups_2_1'];
    embedded_groups_1_3 = scipy.io.loadmat(os.path.join(data_path,'male_embedded_groups_3_1.mat'))['embedded_groups_3_1'];

    chosen_labels_1_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_1_1_is_spoofed.mat'))['chosen_labels_1_1_is_spoofed'];
    chosen_labels_1_1_is_spoofed = pd.Series([item for sublist in chosen_labels_1_1_is_spoofed for item in sublist]);

    chosen_labels_2_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_2_1_is_spoofed.mat'))['chosen_labels_2_1_is_spoofed'];
    chosen_labels_2_1_is_spoofed = pd.Series([item for sublist in chosen_labels_2_1_is_spoofed for item in sublist]);

    chosen_labels_3_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_3_1_is_spoofed.mat'))['chosen_labels_3_1_is_spoofed'];
    chosen_labels_3_1_is_spoofed = pd.Series([item for sublist in chosen_labels_3_1_is_spoofed for item in sublist]);

    chosen_labels_numeric_1_1 = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_numeric_1_1.mat'))['chosen_labels_numeric_1_1'][:];
    chosen_labels_numeric_1_1 = pd.Series([item for sublist in chosen_labels_numeric_1_1 for item in sublist]);

    chosen_labels_numeric_2_1 = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_numeric_2_1.mat'))['chosen_labels_numeric_2_1'][:];
    chosen_labels_numeric_2_1 = pd.Series([item for sublist in chosen_labels_numeric_2_1 for item in sublist]);

    chosen_labels_numeric_3_1 = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_numeric_3_1.mat'))['chosen_labels_numeric_3_1'][:];
    chosen_labels_numeric_3_1 = pd.Series([item for sublist in chosen_labels_numeric_3_1 for item in sublist]);

    chosen_labels_1_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_1_1_attack_logical.mat'))['chosen_labels_1_1_attack_logical'];
    chosen_labels_1_1_attack_logical = pd.Series([item for sublist in chosen_labels_1_1_attack_logical for item in sublist])

    chosen_labels_2_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_2_1_attack_logical.mat'))['chosen_labels_2_1_attack_logical'];
    chosen_labels_2_1_attack_logical = pd.Series([item for sublist in chosen_labels_2_1_attack_logical for item in sublist])

    chosen_labels_3_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_3_1_attack_logical.mat'))['chosen_labels_3_1_attack_logical'];
    chosen_labels_3_1_attack_logical = pd.Series([item for sublist in chosen_labels_3_1_attack_logical for item in sublist])

    chosen_labels_1_1_name = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_1_1_name.mat'))['chosen_labels_1_1_name'];
    chosen_labels_1_1_name = pd.Series([item for sublist in chosen_labels_1_1_name for item in sublist])

    chosen_labels_2_1_name = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_2_1_name.mat'))['chosen_labels_2_1_name'];
    chosen_labels_2_1_name = pd.Series([item for sublist in chosen_labels_2_1_name for item in sublist])

    chosen_labels_3_1_name = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_3_1_name.mat'))['chosen_labels_3_1_name'];
    chosen_labels_3_1_name = pd.Series([item for sublist in chosen_labels_3_1_name for item in sublist])

    chosen_labels_1_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_1_1_speaker_id.mat'))['chosen_labels_1_1_speaker_id'];
    chosen_labels_1_1_speaker_id = pd.Series([item for sublist in chosen_labels_1_1_speaker_id for item in sublist])

    chosen_labels_2_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_2_1_speaker_id.mat'))['chosen_labels_2_1_speaker_id'];
    chosen_labels_2_1_speaker_id = pd.Series([item for sublist in chosen_labels_2_1_speaker_id for item in sublist])

    chosen_labels_3_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_3_1_speaker_id.mat'))['chosen_labels_3_1_speaker_id'];
    chosen_labels_3_1_speaker_id = pd.Series([item for sublist in chosen_labels_3_1_speaker_id for item in sublist])

    male_chosen_labels_1_1_sex = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_1_1_sex.mat'))['chosen_labels_1_1_sex'];
    male_chosen_labels_1_1_sex = pd.Series([item for sublist in male_chosen_labels_1_1_sex for item in sublist]);

    male_chosen_labels_2_1_sex = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_2_1_sex.mat'))['chosen_labels_2_1_sex'];
    male_chosen_labels_2_1_sex = pd.Series([item for sublist in male_chosen_labels_2_1_sex for item in sublist]);

    male_chosen_labels_3_1_sex = scipy.io.loadmat(os.path.join(data_path,'male_chosen_labels_3_1_sex.mat'))['chosen_labels_3_1_sex'];
    male_chosen_labels_3_1_sex = pd.Series([item for sublist in male_chosen_labels_3_1_sex for item in sublist]);
    
    return  embedded_groups_1_1,embedded_groups_1_2,embedded_groups_1_3,chosen_labels_1_1_is_spoofed,chosen_labels_2_1_is_spoofed,chosen_labels_3_1_is_spoofed,chosen_labels_numeric_1_1,chosen_labels_numeric_2_1,chosen_labels_numeric_3_1,chosen_labels_1_1_attack_logical,chosen_labels_2_1_attack_logical,chosen_labels_3_1_attack_logical,chosen_labels_1_1_name,chosen_labels_2_1_name,chosen_labels_3_1_name,chosen_labels_1_1_speaker_id,chosen_labels_2_1_speaker_id,chosen_labels_3_1_speaker_id,male_chosen_labels_1_1_sex,male_chosen_labels_2_1_sex,male_chosen_labels_3_1_sex   

#loading the male data for loading the PMF-based embeddings for female gender in ASVspoof2019
def load_data_female(data_path):
    embedded_groups_1_1 = scipy.io.loadmat(os.path.join(data_path,'female_embedded_groups_1_1.mat'))['embedded_groups_1_1'];
    embedded_groups_1_2 = scipy.io.loadmat(os.path.join(data_path,'female_embedded_groups_2_1.mat'))['embedded_groups_2_1'];
    embedded_groups_1_3 = scipy.io.loadmat(os.path.join(data_path,'female_embedded_groups_3_1.mat'))['embedded_groups_3_1'];

    chosen_labels_1_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_1_1_is_spoofed.mat'))['chosen_labels_1_1_is_spoofed'];
    chosen_labels_1_1_is_spoofed = pd.Series([item for sublist in chosen_labels_1_1_is_spoofed for item in sublist]);

    chosen_labels_2_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_2_1_is_spoofed.mat'))['chosen_labels_2_1_is_spoofed'];
    chosen_labels_2_1_is_spoofed = pd.Series([item for sublist in chosen_labels_2_1_is_spoofed for item in sublist]);

    chosen_labels_3_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_3_1_is_spoofed.mat'))['chosen_labels_3_1_is_spoofed'];
    chosen_labels_3_1_is_spoofed = pd.Series([item for sublist in chosen_labels_3_1_is_spoofed for item in sublist]);

    chosen_labels_numeric_1_1 = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_numeric_1_1.mat'))['chosen_labels_numeric_1_1'][:];
    chosen_labels_numeric_1_1 = pd.Series([item for sublist in chosen_labels_numeric_1_1 for item in sublist]);

    chosen_labels_numeric_2_1 = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_numeric_2_1.mat'))['chosen_labels_numeric_2_1'][:];
    chosen_labels_numeric_2_1 = pd.Series([item for sublist in chosen_labels_numeric_2_1 for item in sublist]);

    chosen_labels_numeric_3_1 = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_numeric_3_1.mat'))['chosen_labels_numeric_3_1'][:];
    chosen_labels_numeric_3_1 = pd.Series([item for sublist in chosen_labels_numeric_3_1 for item in sublist]);

    chosen_labels_1_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_1_1_attack_logical.mat'))['chosen_labels_1_1_attack_logical'];
    chosen_labels_1_1_attack_logical = pd.Series([item for sublist in chosen_labels_1_1_attack_logical for item in sublist])

    chosen_labels_2_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_2_1_attack_logical.mat'))['chosen_labels_2_1_attack_logical'];
    chosen_labels_2_1_attack_logical = pd.Series([item for sublist in chosen_labels_2_1_attack_logical for item in sublist])

    chosen_labels_3_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_3_1_attack_logical.mat'))['chosen_labels_3_1_attack_logical'];
    chosen_labels_3_1_attack_logical = pd.Series([item for sublist in chosen_labels_3_1_attack_logical for item in sublist])

    chosen_labels_1_1_name = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_1_1_name.mat'))['chosen_labels_1_1_name'];
    chosen_labels_1_1_name = pd.Series([item for sublist in chosen_labels_1_1_name for item in sublist])

    chosen_labels_2_1_name = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_2_1_name.mat'))['chosen_labels_2_1_name'];
    chosen_labels_2_1_name = pd.Series([item for sublist in chosen_labels_2_1_name for item in sublist])

    chosen_labels_3_1_name = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_3_1_name.mat'))['chosen_labels_3_1_name'];
    chosen_labels_3_1_name = pd.Series([item for sublist in chosen_labels_3_1_name for item in sublist])

    chosen_labels_1_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_1_1_speaker_id.mat'))['chosen_labels_1_1_speaker_id'];
    chosen_labels_1_1_speaker_id = pd.Series([item for sublist in chosen_labels_1_1_speaker_id for item in sublist])

    chosen_labels_2_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_2_1_speaker_id.mat'))['chosen_labels_2_1_speaker_id'];
    chosen_labels_2_1_speaker_id = pd.Series([item for sublist in chosen_labels_2_1_speaker_id for item in sublist])

    chosen_labels_3_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_3_1_speaker_id.mat'))['chosen_labels_3_1_speaker_id'];
    chosen_labels_3_1_speaker_id = pd.Series([item for sublist in chosen_labels_3_1_speaker_id for item in sublist])

    female_chosen_labels_1_1_sex = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_1_1_sex.mat'))['chosen_labels_1_1_sex'];
    female_chosen_labels_1_1_sex = pd.Series([item for sublist in female_chosen_labels_1_1_sex for item in sublist]);

    female_chosen_labels_2_1_sex = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_2_1_sex.mat'))['chosen_labels_2_1_sex'];
    female_chosen_labels_2_1_sex = pd.Series([item for sublist in female_chosen_labels_2_1_sex for item in sublist]);

    female_chosen_labels_3_1_sex = scipy.io.loadmat(os.path.join(data_path,'female_chosen_labels_3_1_sex.mat'))['chosen_labels_3_1_sex'];
    female_chosen_labels_3_1_sex = pd.Series([item for sublist in female_chosen_labels_3_1_sex for item in sublist]);
    
    return  embedded_groups_1_1,embedded_groups_1_2,embedded_groups_1_3,chosen_labels_1_1_is_spoofed,chosen_labels_2_1_is_spoofed,chosen_labels_3_1_is_spoofed,chosen_labels_numeric_1_1,chosen_labels_numeric_2_1,chosen_labels_numeric_3_1,chosen_labels_1_1_attack_logical,chosen_labels_2_1_attack_logical,chosen_labels_3_1_attack_logical,chosen_labels_1_1_name,chosen_labels_2_1_name,chosen_labels_3_1_name,chosen_labels_1_1_speaker_id,chosen_labels_2_1_speaker_id,chosen_labels_3_1_speaker_id,female_chosen_labels_1_1_sex,female_chosen_labels_2_1_sex,female_chosen_labels_3_1_sex   
