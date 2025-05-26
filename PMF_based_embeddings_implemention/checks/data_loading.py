import os
import scipy
import pandas as pd
import scipy.io
import numpy as np

#loading the female data
def load_data_all(data_path,include_eval=True,include_dev = True):
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
        chosen_labels_3_1_is_spoofed = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_is_spoofed.mat'))['chosen_labels_3_1_is_spoofed'];
        chosen_labels_3_1_is_spoofed = pd.Series([item for sublist in chosen_labels_3_1_is_spoofed for item in sublist]);

    chosen_labels_numeric_1_1 = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_numeric_1_1.mat'))['chosen_labels_numeric_1_1'][:];
    chosen_labels_numeric_1_1 = pd.Series([item for sublist in chosen_labels_numeric_1_1 for item in sublist]);

    chosen_labels_numeric_2_1 = pd.Series([])
    if include_dev:
        chosen_labels_numeric_2_1 = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_numeric_2_1.mat'))['chosen_labels_numeric_2_1'][:];
        chosen_labels_numeric_2_1 = pd.Series([item for sublist in chosen_labels_numeric_2_1 for item in sublist]);
    
    chosen_labels_numeric_3_1 = pd.Series([])
    if include_eval:
        chosen_labels_numeric_3_1 = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_numeric_3_1.mat'))['chosen_labels_numeric_3_1'][:];
        chosen_labels_numeric_3_1 = pd.Series([item for sublist in chosen_labels_numeric_3_1 for item in sublist]);

    chosen_labels_1_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_attack_logical.mat'))['chosen_labels_1_1_attack_logical'];
    chosen_labels_1_1_attack_logical = pd.Series([item for sublist in chosen_labels_1_1_attack_logical for item in sublist])

    chosen_labels_2_1_attack_logical = pd.Series([])
    if include_dev:
        chosen_labels_2_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_attack_logical.mat'))['chosen_labels_2_1_attack_logical'];
        chosen_labels_2_1_attack_logical = pd.Series([item for sublist in chosen_labels_2_1_attack_logical for item in sublist])

    chosen_labels_3_1_attack_logical = pd.Series([])
    if include_eval:
        chosen_labels_3_1_attack_logical = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_attack_logical.mat'))['chosen_labels_3_1_attack_logical'];
        chosen_labels_3_1_attack_logical = pd.Series([item for sublist in chosen_labels_3_1_attack_logical for item in sublist])

    chosen_labels_1_1_name = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_name.mat'))['chosen_labels_1_1_name'];
    chosen_labels_1_1_name = pd.Series([item for sublist in chosen_labels_1_1_name for item in sublist])

    chosen_labels_2_1_name = pd.Series([])
    if include_dev:
        chosen_labels_2_1_name = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_name.mat'))['chosen_labels_2_1_name'];
        chosen_labels_2_1_name = pd.Series([item for sublist in chosen_labels_2_1_name for item in sublist])
    
    chosen_labels_3_1_name = pd.Series([])
    if include_eval:
        chosen_labels_3_1_name = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_name.mat'))['chosen_labels_3_1_name'];
        chosen_labels_3_1_name = pd.Series([item for sublist in chosen_labels_3_1_name for item in sublist])

    chosen_labels_1_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_speaker_id.mat'))['chosen_labels_1_1_speaker_id'];
    chosen_labels_1_1_speaker_id = pd.Series([item for sublist in chosen_labels_1_1_speaker_id for item in sublist])

    chosen_labels_2_1_speaker_id = pd.Series([])
    if include_dev:
        chosen_labels_2_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_speaker_id.mat'))['chosen_labels_2_1_speaker_id'];
        chosen_labels_2_1_speaker_id = pd.Series([item for sublist in chosen_labels_2_1_speaker_id for item in sublist])
    
    chosen_labels_3_1_speaker_id = pd.Series([])
    if include_eval:
        chosen_labels_3_1_speaker_id = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_speaker_id.mat'))['chosen_labels_3_1_speaker_id'];
        chosen_labels_3_1_speaker_id = pd.Series([item for sublist in chosen_labels_3_1_speaker_id for item in sublist])

    chosen_labels_1_1_sex = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_1_1_sex.mat'))['chosen_labels_1_1_sex'];
    chosen_labels_1_1_sex = pd.Series([item for sublist in chosen_labels_1_1_sex for item in sublist]);

    chosen_labels_2_1_sex = pd.Series([])
    if include_dev:
        chosen_labels_2_1_sex = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_2_1_sex.mat'))['chosen_labels_2_1_sex'];
        chosen_labels_2_1_sex = pd.Series([item for sublist in chosen_labels_2_1_sex for item in sublist]);
    
    chosen_labels_3_1_sex = pd.Series([])
    if include_eval:
        chosen_labels_3_1_sex = scipy.io.loadmat(os.path.join(data_path,'all_chosen_labels_3_1_sex.mat'))['chosen_labels_3_1_sex'];
        chosen_labels_3_1_sex = pd.Series([item for sublist in chosen_labels_3_1_sex for item in sublist]);
    
    
    return  embedded_groups_1_1,embedded_groups_1_2,embedded_groups_1_3,chosen_labels_1_1_is_spoofed,chosen_labels_2_1_is_spoofed,chosen_labels_3_1_is_spoofed,chosen_labels_numeric_1_1,chosen_labels_numeric_2_1,chosen_labels_numeric_3_1,chosen_labels_1_1_attack_logical,chosen_labels_2_1_attack_logical,chosen_labels_3_1_attack_logical,chosen_labels_1_1_name,chosen_labels_2_1_name,chosen_labels_3_1_name,chosen_labels_1_1_speaker_id,chosen_labels_2_1_speaker_id,chosen_labels_3_1_speaker_id,chosen_labels_1_1_sex,chosen_labels_2_1_sex,chosen_labels_3_1_sex   



#loading the male data
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

#loading the female data
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




#concatenate the data of male and female to one data
def concatenate_data(male_embedded_groups_1_1,male_embedded_groups_1_2,male_embedded_groups_1_3,
                male_chosen_labels_1_1_is_spoofed,male_chosen_labels_2_1_is_spoofed,male_chosen_labels_3_1_is_spoofed,
                male_chosen_labels_numeric_1_1,male_chosen_labels_numeric_2_1,male_chosen_labels_numeric_3_1,
                male_chosen_labels_1_1_attack_logical,male_chosen_labels_2_1_attack_logical,male_chosen_labels_3_1_attack_logical,
                male_chosen_labels_1_1_name,male_chosen_labels_2_1_name,male_chosen_labels_3_1_name,
                male_chosen_labels_1_1_speaker_id,male_chosen_labels_2_1_speaker_id, male_chosen_labels_3_1_speaker_id,
                male_chosen_labels_1_1_sex,male_chosen_labels_2_1_sex,male_chosen_labels_3_1_sex,
                
                female_embedded_groups_1_1,female_embedded_groups_1_2,female_embedded_groups_1_3,
                female_chosen_labels_1_1_is_spoofed,female_chosen_labels_2_1_is_spoofed,female_chosen_labels_3_1_is_spoofed,
                female_chosen_labels_numeric_1_1,female_chosen_labels_numeric_2_1,female_chosen_labels_numeric_3_1,
                female_chosen_labels_1_1_attack_logical,female_chosen_labels_2_1_attack_logical,female_chosen_labels_3_1_attack_logical,
                female_chosen_labels_1_1_name,female_chosen_labels_2_1_name,female_chosen_labels_3_1_name,
                female_chosen_labels_1_1_speaker_id,female_chosen_labels_2_1_speaker_id,female_chosen_labels_3_1_speaker_id,
                female_chosen_labels_1_1_sex,female_chosen_labels_2_1_sex,female_chosen_labels_3_1_sex):


    embedded_groups_1_1 = np.concatenate((male_embedded_groups_1_1,female_embedded_groups_1_1),axis = 0); 

    embedded_groups_1_2 = np.concatenate((male_embedded_groups_1_2,female_embedded_groups_1_2),axis = 0);

    embedded_groups_1_3 = np.concatenate((male_embedded_groups_1_3,female_embedded_groups_1_3),axis = 0);

    chosen_labels_numeric_1_1 = pd.concat([male_chosen_labels_numeric_1_1,female_chosen_labels_numeric_1_1], ignore_index = True,axis = 0);
    chosen_labels_numeric_2_1 = pd.concat([male_chosen_labels_numeric_2_1,female_chosen_labels_numeric_2_1], ignore_index = True,axis = 0);
    chosen_labels_numeric_3_1 = pd.concat([male_chosen_labels_numeric_3_1,female_chosen_labels_numeric_3_1], ignore_index = True, axis = 0);
    
    chosen_labels_1_1_attack_logical = pd.concat([male_chosen_labels_1_1_attack_logical,female_chosen_labels_1_1_attack_logical], ignore_index = True,axis = 0);
    chosen_labels_1_1_is_spoofed = pd.concat([male_chosen_labels_1_1_is_spoofed,female_chosen_labels_1_1_is_spoofed], ignore_index = True,axis = 0);
    chosen_labels_1_1_name = pd.concat([male_chosen_labels_1_1_name,female_chosen_labels_1_1_name], ignore_index = True, axis = 0);
    chosen_labels_1_1_sex = pd.concat([male_chosen_labels_1_1_sex,female_chosen_labels_1_1_sex], ignore_index = True,axis = 0);
    chosen_labels_1_1_speaker_id = pd.concat([male_chosen_labels_1_1_speaker_id,female_chosen_labels_1_1_speaker_id], ignore_index = True, axis = 0);

    chosen_labels_2_1_attack_logical = pd.concat([male_chosen_labels_2_1_attack_logical,female_chosen_labels_2_1_attack_logical], ignore_index = True,axis = 0);
    chosen_labels_2_1_is_spoofed = pd.concat([male_chosen_labels_2_1_is_spoofed,female_chosen_labels_2_1_is_spoofed], ignore_index = True,axis = 0);
    chosen_labels_2_1_name = pd.concat([male_chosen_labels_2_1_name,female_chosen_labels_2_1_name], ignore_index = True, axis = 0);
    chosen_labels_2_1_sex = pd.concat([male_chosen_labels_2_1_sex,female_chosen_labels_2_1_sex], ignore_index = True,axis = 0);
    chosen_labels_2_1_speaker_id = pd.concat([male_chosen_labels_2_1_speaker_id,female_chosen_labels_2_1_speaker_id], ignore_index = True, axis = 0);

    chosen_labels_3_1_attack_logical = pd.concat([male_chosen_labels_3_1_attack_logical,female_chosen_labels_3_1_attack_logical], ignore_index = True,axis = 0);
    chosen_labels_3_1_is_spoofed = pd.concat([male_chosen_labels_3_1_is_spoofed,female_chosen_labels_3_1_is_spoofed], ignore_index = True,axis = 0);
    chosen_labels_3_1_name = pd.concat([male_chosen_labels_3_1_name,female_chosen_labels_3_1_name], ignore_index = True, axis = 0);
    chosen_labels_3_1_sex = pd.concat([male_chosen_labels_3_1_sex,female_chosen_labels_3_1_sex], ignore_index = True,axis = 0);
    chosen_labels_3_1_speaker_id = pd.concat([male_chosen_labels_3_1_speaker_id,female_chosen_labels_3_1_speaker_id], ignore_index = True, axis = 0);
    
    return  embedded_groups_1_1,embedded_groups_1_2,embedded_groups_1_3,chosen_labels_1_1_is_spoofed,chosen_labels_2_1_is_spoofed,chosen_labels_3_1_is_spoofed,chosen_labels_numeric_1_1,chosen_labels_numeric_2_1,chosen_labels_numeric_3_1,chosen_labels_1_1_attack_logical,chosen_labels_2_1_attack_logical,chosen_labels_3_1_attack_logical,chosen_labels_1_1_name,chosen_labels_2_1_name,chosen_labels_3_1_name,chosen_labels_1_1_speaker_id,chosen_labels_2_1_speaker_id,chosen_labels_3_1_speaker_id,chosen_labels_1_1_sex,chosen_labels_2_1_sex,chosen_labels_3_1_sex   


#function to get the real channel number                        
def get_real_channel(channels,num_of_channels):
    if num_of_channels <= 1:
        real_channel = 0
        return real_channel
    
    real_channel = []
    for i in range(len(channels)):
        real_channel.append(channels[i] - num_of_channels/2);
        if real_channel[i] <= 0:
            real_channel[i] = real_channel[i] + num_of_channels
    return real_channel



#function for XAI for RFC to see the feature importance
def get_columns_names_feature_importance(substruct = True): #substruct = True means that we want to substruct the distance from spoof and human
    number_filters = 2

    filter_names = ["gammtone_inv","gammatone"] #filters names 
   # filter_names = ["gammatone_inv","gammatone"]
    number_channels = 10 #number of channels    

    distance_from_spoof_and_human = 2 #distance from spoof and human
    distance_from_spoof_and_human_names = ["d_(p,p_h)","d_(p,p_s)"] #distance from spoof and human names
    number_of_distances = 8 #number of distances
    distance_names = ["Chi-square","Correlation","Hellinger","Intersection","Jensen-Shannon","Symmetrised Kullback-Leibler","Kullback-Leibler Divergence","Modified Kolmogorov-Smirnov"] #distance names

    columns_names = [] #columns names

    max_name_length = number_filters*number_channels*number_of_distances*distance_from_spoof_and_human #max name length


    #for loop to create the columns names when substruct = True 
    if substruct == True:
        for i in range(0,number_filters):
            for j in range(0,number_channels):
                for k in range(0,number_of_distances):
                        #print(f"filter-{filter_names[i]}-channel-{j+1}-distance-{distance_names[k]}-[d_(p,p_s)-d(p,p_h)] - {max_name_length//2}")
                            columns_names.append(f"filter-{filter_names[i]}-channel-{j+1}-distance-{distance_names[k]}-[d_(p,p_s)-d(p,p_h)]")


    #for loop to create the columns names when substruct = False
    if substruct == False:
        for i in range(0,number_filters):
            for j in range(0,number_channels):
                    for l in range(0,distance_from_spoof_and_human):
                        for k in range(0,number_of_distances):
                            #print(f"filter{filter_names[i]}-channel-{j}-distance-{distance_names[k]}-{distance_from_spoof_and_human_names[l]} - {max_name_length}")
                            columns_names.append(f"filter{filter_names[i]}-channel-{j+1}-distance-{distance_names[k]}-{distance_from_spoof_and_human_names[l]}")
                            
    return columns_names,max_name_length