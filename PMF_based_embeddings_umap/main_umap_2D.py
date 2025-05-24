import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from loading_time_embeddings_datasets import load_data_female, load_data_male , load_data_all_ASVSpoof5 , read_ASVSpoof5_protocol
from plots_function import plotting_genuine, plotting_genuine_by_without_codec
from umap import UMAP, ParametricUMAP
from umap_class import class_time_embeddings_umap



if __name__ == "__main__":
    # Set working directory
    os.chdir('./PMF_based_embeddings_umap') 
    asvspoof5_train_protocol = read_ASVSpoof5_protocol('./ASVspoof5_protocols/ASVspoof5.train.tsv')
    asvspoof5_dev_protocol = read_ASVSpoof5_protocol('./ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv')
    asvspoof5_eval_protocol = read_ASVSpoof5_protocol('./ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv')

    
    # We recommend using the UMAP model trained on time embeddings from the ASVspoof2019 dataset.
    # The model was created using the class defined in class_time_embeddings_umap.py.
    # Now, simply load the recommended pre-trained UMAP model from the file 'time_emb.pkl'.
    time_emb = pickle.load(open('./time_emb.pkl', 'rb'))
    
    # ============== ASVspoof2019 ==============
    
    # === plot the genuine time embeddings of the training set ===
    
    plotting_genuine(time_emb.umap_train, time_emb.embedded_groups_1_1, time_emb.chosen_labels_1_1_sex, time_emb.chosen_labels_1_1_attack_logical_mapping, \
        plot_title='UMAP - Train on Trn. ASVSpoof2019 \n Time Embeddings Trn. ASVSpoof2019', gender='both', frontsize=20)

    # === plot the genuine time embeddings of the Dev. set ===
    
    plotting_genuine(time_emb.umap_train, time_emb.embedded_groups_1_2, time_emb.chosen_labels_2_1_sex, time_emb.chosen_labels_2_1_attack_logical_mapping, \
        plot_title='UMAP - Train on Trn. ASVSpoof2019 \n Time Embeddings Dev. ASVSpoof2019', gender='both', frontsize=20)
    
    # === plot the genuine time embeddings of the Eval. set ===

    plotting_genuine(time_emb.umap_train, time_emb.embedded_groups_1_3, time_emb.chosen_labels_3_1_sex, time_emb.chosen_labels_3_1_attack_logical_mapping, \
        plot_title='UMAP - Train on Trn. ASVSpoof2019 \n Time Embeddings Eval. ASVSpoof2019', gender='both', frontsize=20)

    # ============== ASVspoof5 ============== 

    # === plot the genuine time embeddings of the training set ===
    
    plotting_genuine(time_emb.umap_train, time_emb.embedded_groups_1_1_asvspoof5, time_emb.chosen_labels_1_1_sex_asvspoof5, time_emb.chosen_labels_1_1_attack_logical_mapping_asvspoof5, \
        plot_title='UMAP - Train on Trn. ASVSpoof2019 \n Time Embeddings Trn. ASVSpoof5', gender='both', frontsize=20)
    
    # === plot the genuine time embeddings of the Dev. set ===

    plotting_genuine(time_emb.umap_train, time_emb.embedded_groups_1_2_asvspoof5, time_emb.chosen_labels_2_1_sex_asvspoof5, time_emb.chosen_labels_2_1_attack_logical_mapping_asvspoof5, \
        plot_title='UMAP - Train on Trn. ASVSpoof2019 \n Time Embeddings Dev. ASVSpoof5', gender='both', frontsize=20)
    
    # === plot the genuine time embeddings of the Eval. set ===

    plotting_genuine(time_emb.umap_train, time_emb.embedded_groups_1_3_asvspoof5, time_emb.chosen_labels_3_1_sex_asvspoof5, time_emb.chosen_labels_3_1_attack_logical_mapping_asvspoof5, \
        plot_title='UMAP - Train on Trn. ASVSpoof2019 \n Time Embeddings Eval. ASVSpoof5', gender='both', frontsize=20)

    # === plot the genuine time embeddings of the Eval. set by without the codecs ===

    plotting_genuine_by_without_codec(time_emb.umap_train, time_emb.embedded_groups_1_3_asvspoof5, time_emb.chosen_labels_3_1_sex_asvspoof5,\
        time_emb.chosen_labels_3_1_attack_logical_mapping_asvspoof5, \
        asvspoof5_eval_protocol['CODEC'], asvspoof5_eval_protocol['CODEC'].unique(),\
        plot_title='UMAP - Train on ASVSpoof2019 \n Time Embeddings Eval. ASVSpoof5', gender='both', frontsize=20)


