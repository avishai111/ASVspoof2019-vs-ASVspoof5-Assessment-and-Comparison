import sys
import os
import torch
import sklearn  
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from loading_time_embeddings_datasets import load_data_female, load_data_male , load_data_all_ASVSpoof5 , read_ASVSpoof5_protocol
from plots_function import plotting_genuine, plotting_genuine_by_without_codec
from umap import UMAP, ParametricUMAP


# This class is used to create UMAP embeddings for PMF-based embeddings.
class class_time_embeddings_umap:
    def __init__(self,path_to_male_embeddings, path_to_female_embeddings):
        self.path_to_male_embeddings = path_to_male_embeddings
        self.path_to_female_embeddings = path_to_female_embeddings
        
        (embedded_groups_1_1_male, embedded_groups_1_2_male, embedded_groups_1_3_male,
            chosen_labels_1_1_is_spoofed_male, chosen_labels_2_1_is_spoofed_male, chosen_labels_3_1_is_spoofed_male,
            chosen_labels_numeric_1_1_male, chosen_labels_numeric_2_1_male, chosen_labels_numeric_3_1_male,
            chosen_labels_1_1_attack_logical_male, chosen_labels_2_1_attack_logical_male, chosen_labels_3_1_attack_logical_male,
            chosen_labels_1_1_name_male, chosen_labels_2_1_name_male, chosen_labels_3_1_name_male,
            chosen_labels_1_1_speaker_id_male, chosen_labels_2_1_speaker_id_male, chosen_labels_3_1_speaker_id_male,
            chosen_labels_1_1_sex_male, chosen_labels_2_1_sex_male, chosen_labels_3_1_sex_male) = load_data_male(self.path_to_male_embeddings)

        (embedded_groups_1_1_female, embedded_groups_1_2_female, embedded_groups_1_3_female,
            chosen_labels_1_1_is_spoofed_female, chosen_labels_2_1_is_spoofed_female, chosen_labels_3_1_is_spoofed_female,
            chosen_labels_numeric_1_1_female, chosen_labels_numeric_2_1_female, chosen_labels_numeric_3_1_female,
            chosen_labels_1_1_attack_logical_female, chosen_labels_2_1_attack_logical_female, chosen_labels_3_1_attack_logical_female,
            chosen_labels_1_1_name_female, chosen_labels_2_1_name_female, chosen_labels_3_1_name_female,
            chosen_labels_1_1_speaker_id_female, chosen_labels_2_1_speaker_id_female, chosen_labels_3_1_speaker_id_female,
            chosen_labels_1_1_sex_female, chosen_labels_2_1_sex_female, chosen_labels_3_1_sex_female) = load_data_female(self.path_to_female_embeddings)
        
        print('ASVspoof2019 data loaded')
        
        embedded_groups_1_1_asvspoof5,embedded_groups_1_2_asvspoof5,embedded_groups_1_3_asvspoof5,chosen_labels_1_1_is_spoofed_asvspoof5,chosen_labels_2_1_is_spoofed_asvspoof5, \
        chosen_labels_3_1_is_spoofed_asvspoof5,chosen_labels_numeric_1_1_asvspoof5,chosen_labels_numeric_2_1_asvspoof5,chosen_labels_numeric_3_1_asvspoof5, \
        chosen_labels_1_1_attack_logical_asvspoof5,chosen_labels_2_1_attack_logical_asvspoof5,chosen_labels_3_1_attack_logical_asvspoof5, \
        chosen_labels_1_1_name_asvspoof5,chosen_labels_2_1_name_asvspoof5,chosen_labels_3_1_name_asvspoof5,chosen_labels_1_1_speaker_id_asvspoof5, \
        chosen_labels_2_1_speaker_id_asvspoof5,chosen_labels_3_1_speaker_id_asvspoof5, \
        chosen_labels_1_1_sex_asvspoof5,chosen_labels_2_1_sex_asvspoof5,chosen_labels_3_1_sex_asvspoof5 = load_data_all_ASVSpoof5('./Data/ASVSpoof5_all_Time_Embeddings/',
                                                                                            include_eval=True,include_dev = True)
        
        print('ASVspoof5 data loaded')
        
        
        
        self.embedded_groups_1_1_asvspoof5 = embedded_groups_1_1_asvspoof5
        self.embedded_groups_1_2_asvspoof5 = embedded_groups_1_2_asvspoof5
        self.embedded_groups_1_3_asvspoof5 = embedded_groups_1_3_asvspoof5
        self.chosen_labels_1_1_is_spoofed_asvspoof5 = chosen_labels_1_1_is_spoofed_asvspoof5
        self.chosen_labels_2_1_is_spoofed_asvspoof5 = chosen_labels_2_1_is_spoofed_asvspoof5
        self.chosen_labels_3_1_is_spoofed_asvspoof5 = chosen_labels_3_1_is_spoofed_asvspoof5


        self.train_embedding_pca = None
        self.dev_embedding_pca = None
        self.train_embedding_umap = None
        self.dev_embedding_umap = None
        
        self.chosen_labels_1_1_attack_logical_asvspoof5 = chosen_labels_1_1_attack_logical_asvspoof5
        self.chosen_labels_2_1_attack_logical_asvspoof5 = chosen_labels_2_1_attack_logical_asvspoof5
        self.chosen_labels_3_1_attack_logical_asvspoof5 = chosen_labels_3_1_attack_logical_asvspoof5
        
        self.chosen_labels_1_1_sex_asvspoof5 = chosen_labels_1_1_sex_asvspoof5
        self.chosen_labels_2_1_sex_asvspoof5 = chosen_labels_2_1_sex_asvspoof5
        self.chosen_labels_3_1_sex_asvspoof5 = chosen_labels_3_1_sex_asvspoof5
    
        
        
        self.chosen_labels_1_1_attack_logical_asvspoof5 = pd.Series([x[0] for x in self.chosen_labels_1_1_attack_logical_asvspoof5]).replace('none', 'Genuine').replace('bonafide', 'Genuine')  
        
        self.chosen_labels_2_1_attack_logical_asvspoof5 = pd.Series([x[0] for x in self.chosen_labels_2_1_attack_logical_asvspoof5]).replace('none', 'Genuine').replace('bonafide', 'Genuine')  
        
        self.chosen_labels_3_1_attack_logical_asvspoof5 = pd.Series([x[0] for x in self.chosen_labels_3_1_attack_logical_asvspoof5]).replace('none', 'Genuine').replace('bonafide', 'Genuine')  
        
        self.chosen_labels_1_1_sex_asvspoof5 = pd.Series([x[0] for x in self.chosen_labels_1_1_sex_asvspoof5])
        
        self.chosen_labels_2_1_sex_asvspoof5 = pd.Series([x[0] for x in self.chosen_labels_2_1_sex_asvspoof5])
        
        self.chosen_labels_3_1_sex_asvspoof5 = pd.Series([x[0] for x in self.chosen_labels_3_1_sex_asvspoof5])
        
        # Generate a colormap
         
        colors_asvspoof5 = np.array([
                    "#A52A2A",  # Brown
                    "#FFD700",  # Gold
                    "#2E8B57",  # SeaGreen
                    "#800080",  # Purple
                    "#FFFF00",  # Yellow
                    "#00FF00",  # Green
                    "#FF00FF",  # Magenta
                    "#800000",  # Maroon
                    "#808000",  # Olive
                    "#800080",  # Purple (Repeated in your list)
                    "#008080",  # Teal
                    "#7FFF00",  # Chartreuse
                    "#D2691E",  # Chocolate
                    "#DC143C",  # Crimson
                    "#4B0082",  # Indigo
                    "#ADFF2F",  # GreenYellow
                    "#FF4500",  # OrangeRed
                    "#FF6347",  # Tomato
                    "#FF8C00",  # DarkOrange
                    "#FFA07A",  # LightSalmon
                    "#FAA568",  # Orange
                    "#4682B4",  # SteelBlue
                    "#5F9EA0",  # CadetBlue
                    "#6495ED",  # CornflowerBlue
                    "#7B68EE",  # MediumSlateBlue
                    "#00CED1",  # DarkTurquoise
                    "#20B2AA",  # LightSeaGreen
                    "#8A2BE2",  # BlueViolet
                    "#B22222",  # Firebrick
                    "#32CD32",  # LimeGreen
                    "#FF1493",  # DeepPink
                    "#1E90FF",  # DodgerBlue
                    "#40E0D0",  # Turquoise
                    "#F08080",  # LightCoral
                ])

       
        labels_asvspoof5 = [f"A{str(i).zfill(2)}" for i in range(1, 33) if i not in [15, 17, 18]]

        # Add the 'none' label
        labels_asvspoof5.append('Genuine')

        # Create the dictionary mapping labels to colors
        label_to_color_asvspoof5 = {label: color for label, color in zip(labels_asvspoof5, colors_asvspoof5)}

        # Explicitly set 'none' to BLUE color
        label_to_color_asvspoof5['Genuine'] = "#0000FF" # Blue
        label_to_color_asvspoof5['A18'] = "#00FFFF" # Cyan
        label_to_color_asvspoof5['A17'] = "#FF0000" # Red
        label_to_color_asvspoof5['A15'] = "#FFA500" # Orange
        

        # Select the keys to extract
        keys_to_extract_asvspoof5 = ['Genuine', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08']
        selected_colors_A01_A08_asvspoof5 = {key: label_to_color_asvspoof5[key] for key in keys_to_extract_asvspoof5}
        
        
        keys_to_extract_2_asvspoof5 = ['Genuine', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
        selected_colors_A09_A16_asvspoof5 = {key: label_to_color_asvspoof5[key] for key in keys_to_extract_2_asvspoof5}

        #'A01', 'A02', 'A03', 'A04', 'A05', 'A06',
        keys_to_extract_3_asvspoof5 = ['Genuine', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 
                             'A29', 'A30', 'A31', 'A32']
        selected_colors_A16_A32_asvspoof5 = {key: label_to_color_asvspoof5[key] for key in keys_to_extract_3_asvspoof5}

        
        self.label_to_color_1_asvspoof5 = selected_colors_A01_A08_asvspoof5
        
        self.label_to_color_2_asvspoof5 = selected_colors_A09_A16_asvspoof5
        
        self.labels_to_color_3_asvspoof5 = selected_colors_A16_A32_asvspoof5
        
        self.chosen_labels_1_1_attack_logical_mapping_asvspoof5 = self.chosen_labels_1_1_attack_logical_asvspoof5.map(self.label_to_color_1_asvspoof5)
       
        self.chosen_labels_2_1_attack_logical_mapping_asvspoof5 = self.chosen_labels_2_1_attack_logical_asvspoof5.map(self.label_to_color_2_asvspoof5)
        
        self.chosen_labels_3_1_attack_logical_mapping_asvspoof5 = self.chosen_labels_3_1_attack_logical_asvspoof5.map(self.labels_to_color_3_asvspoof5)
        
        # Stack male and female data
        self.embedded_groups_1_1 = np.vstack((embedded_groups_1_1_male, embedded_groups_1_1_female))
        self.embedded_groups_1_2 = np.vstack((embedded_groups_1_2_male, embedded_groups_1_2_female))
        self.embedded_groups_1_3 = np.vstack((embedded_groups_1_3_male, embedded_groups_1_3_female))

        self.chosen_labels_1_1_is_spoofed = np.hstack((chosen_labels_1_1_is_spoofed_male, chosen_labels_1_1_is_spoofed_female))
        self.chosen_labels_2_1_is_spoofed = np.hstack((chosen_labels_2_1_is_spoofed_male, chosen_labels_2_1_is_spoofed_female))
        self.chosen_labels_3_1_is_spoofed = np.hstack((chosen_labels_3_1_is_spoofed_male, chosen_labels_3_1_is_spoofed_female))

        self.chosen_labels_numeric_1_1 = np.hstack((chosen_labels_numeric_1_1_male, chosen_labels_numeric_1_1_female))
        self.chosen_labels_numeric_2_1 = np.hstack((chosen_labels_numeric_2_1_male, chosen_labels_numeric_2_1_female))
        self.chosen_labels_numeric_3_1 = np.hstack((chosen_labels_numeric_3_1_male, chosen_labels_numeric_3_1_female))

        self.chosen_labels_1_1_attack_logical = pd.concat([pd.Series(chosen_labels_1_1_attack_logical_male),
                                                           pd.Series(chosen_labels_1_1_attack_logical_female)])
        self.chosen_labels_2_1_attack_logical = pd.concat([pd.Series(chosen_labels_2_1_attack_logical_male),
                                                           pd.Series(chosen_labels_2_1_attack_logical_female)])
        self.chosen_labels_3_1_attack_logical = pd.concat([pd.Series(chosen_labels_3_1_attack_logical_male),
                                                           pd.Series(chosen_labels_3_1_attack_logical_female)])

        self.chosen_labels_1_1_name = np.hstack((chosen_labels_1_1_name_male, chosen_labels_1_1_name_female))
        self.chosen_labels_2_1_name = np.hstack((chosen_labels_2_1_name_male, chosen_labels_2_1_name_female))
        self.chosen_labels_3_1_name = np.hstack((chosen_labels_3_1_name_male, chosen_labels_3_1_name_female))

        self.chosen_labels_1_1_speaker_id = np.hstack((chosen_labels_1_1_speaker_id_male, chosen_labels_1_1_speaker_id_female))
        self.chosen_labels_2_1_speaker_id = np.hstack((chosen_labels_2_1_speaker_id_male, chosen_labels_2_1_speaker_id_female))
        self.chosen_labels_3_1_speaker_id = np.hstack((chosen_labels_3_1_speaker_id_male, chosen_labels_3_1_speaker_id_female))

        self.chosen_labels_1_1_sex = pd.concat([pd.Series(chosen_labels_1_1_sex_male), pd.Series(chosen_labels_1_1_sex_female)])
        self.chosen_labels_2_1_sex = pd.concat([pd.Series(chosen_labels_2_1_sex_male), pd.Series(chosen_labels_2_1_sex_female)])
        self.chosen_labels_3_1_sex = pd.concat([pd.Series(chosen_labels_3_1_sex_male), pd.Series(chosen_labels_3_1_sex_female)])
        
        
        self.train_embedding_pca = None
        self.dev_embedding_pca = None
        self.train_embedding_umap = None
        self.dev_embedding_umap = None
        
        self.chosen_labels_1_1_attack_logical = pd.Series([x[0] for x in self.chosen_labels_1_1_attack_logical]).replace('none', 'Genuine')
        
        self.chosen_labels_2_1_attack_logical = pd.Series([x[0] for x in self.chosen_labels_2_1_attack_logical]).replace('none', 'Genuine')
        
        self.chosen_labels_3_1_attack_logical = pd.Series([x[0] for x in self.chosen_labels_3_1_attack_logical]).replace('none', 'Genuine')
        
        self.chosen_labels_1_1_sex = pd.Series([x[0] for x in self.chosen_labels_1_1_sex])
        
        self.chosen_labels_2_1_sex = pd.Series([x[0] for x in self.chosen_labels_2_1_sex])
        
        self.chosen_labels_3_1_sex = pd.Series([x[0] for x in self.chosen_labels_3_1_sex])
        
        
        # Define unique labels including 'genuine'
        unique_labels = np.concatenate([self.chosen_labels_1_1_attack_logical.unique() , self.chosen_labels_2_1_attack_logical.unique()], axis=0)
        
        
        # Generate a colormap
         
            
        colors = np.array([
            "#A52A2A",  # Brown
            "#FFD700",  # Gold
            "#2E8B57",  # SeaGreen
            "#800080",  # Purple
            "#FFFF00",  # Yellow
            "#00FF00",  # Green
            "#FF00FF",  # Magenta
            "#800000",  # Maroon
            "#808000",  # Olive
            "#800080",  # Purple
            "#008080",  # Teal
            "#7FFF00",  # Chartreuse
            "#D2691E",  # Chocolate
            "#DC143C",  # Crimson
            "#4B0082",  # Indigo
            "#ADFF2F",  # GreenYellow
            "#FF4500",  # OrangeRed
        ])

        labels = [f"A{str(i).zfill(2)}" for i in range(1, 20) if i not in [15, 17, 18]]

        # Add the 'none' label
        labels.append('Genuine')

        # Create the dictionary mapping labels to colors
        label_to_color = {label: color for label, color in zip(labels, colors)}

        # Explicitly set 'none' to BLUE color
        label_to_color['Genuine'] = "#0000FF" # Blue
        label_to_color['A18'] = "#00FFFF" # Cyan
        label_to_color['A17'] = "#FF0000" # Red
        label_to_color['A15'] = "#FFA500" # Orange
        

        # Select the keys to extract
        keys_to_extract = ['Genuine', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']
        selected_colors_A01_A06 = {key: label_to_color[key] for key in keys_to_extract}

        #'A01', 'A02', 'A03', 'A04', 'A05', 'A06',
        keys_to_extract_2 = ['Genuine', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']
        selected_colors_A01_A19 = {key: label_to_color[key] for key in keys_to_extract_2}

        
        self.label_to_color_1 = selected_colors_A01_A06
        
        self.labels_to_color_3 = selected_colors_A01_A19
        
        
        self.chosen_labels_1_1_attack_logical_mapping = self.chosen_labels_1_1_attack_logical.map(self.label_to_color_1)
       
        self.chosen_labels_2_1_attack_logical_mapping = self.chosen_labels_2_1_attack_logical.map(self.label_to_color_1)
        
        self.chosen_labels_3_1_attack_logical_mapping = self.chosen_labels_3_1_attack_logical.map(self.labels_to_color_3)
        
        self.frontsize = 14
        
    def create_umap(self,include_dev=False,include_eval=False):
        
        print('Training UMAP model..')
        
        self.umap_train = UMAP(n_components=2,random_state=10,n_jobs = 1) # to make this consistent, we can set the random state to a fixed value.
        self.umap_train.fit(self.embedded_groups_1_1)
        
        print('UMAP model trained')
        
        if include_dev:
    
            self.umap_dev = self.umap_train
        
        if include_eval:

            self.umap_eval = self.umap_train
       
    