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
from umap_class import class_time_embeddings_umap , save_npz_file



if __name__ == "__main__":
    # Set working directory
    # os.chdir('./PMF_based_embeddings_umap') 
    asvspoof5_train_protocol = read_ASVSpoof5_protocol('./ASVspoof5_protocols/ASVspoof5.train.tsv')
    asvspoof5_dev_protocol = read_ASVSpoof5_protocol('./ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv')
    asvspoof5_eval_protocol = read_ASVSpoof5_protocol('./ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv')

    
    # We are using the UMAP model trained on time embeddings from the training set in ASVspoof2019 database.
    # The class of the UMAP model was created using the class defined in class_time_embeddings_umap.py,
    # initially saved as a .pkl file and later converted to a .npz file for easier data loading.
    # Now, we simply load the .npz file containing the time embeddings transformed into 2D using UMAP algorithm.
    time_emb = np.load("time_embeddings_data.npz", allow_pickle=True)
    

    # ============== ASVspoof2019 ==============
    
    # === plot the genuine time embeddings of the training set ===
    
    plotting_genuine(time_emb["time_embed_train_2019_samples_2D"], time_emb["chosen_labels_1_1_sex"], time_emb["chosen_labels_1_1_attack_logical_mapping"], \
        plot_title='UMAP - Train on Trn. ASVSpoof2019 \n Time Embeddings Trn. ASVSpoof2019', gender='both', frontsize=20)

    # === plot the genuine time embeddings of the Dev. set ===
    
    plotting_genuine(time_emb["time_embed_dev_2019_samples_2D"], time_emb["chosen_labels_2_1_sex"], time_emb["chosen_labels_2_1_attack_logical_mapping"], \
        plot_title='UMAP - Train on Trn. ASVSpoof2019 \n Time Embeddings Dev. ASVSpoof2019', gender='both', frontsize=20)
    
    # === plot the genuine time embeddings of the Eval. set ===

    plotting_genuine(time_emb["time_embed_eval_2019_samples_2D"], time_emb["chosen_labels_3_1_sex"], time_emb["chosen_labels_3_1_attack_logical_mapping"], \
        plot_title='UMAP - Train on Trn. ASVSpoof2019 \n Time Embeddings Eval. ASVSpoof2019', gender='both', frontsize=20)

    # ============== ASVspoof5 ============== 

    # === plot the genuine time embeddings of the training set ===
    
    plotting_genuine(time_emb["time_embed_train_05_samples_2D"], time_emb["chosen_labels_1_1_sex_asvspoof5"], time_emb["chosen_labels_1_1_attack_logical_mapping_asvspoof5"], \
        plot_title='UMAP - Train on Trn. ASVSpoof2019 \n Time Embeddings Trn. ASVSpoof5', gender='both', frontsize=20)
    
    # === plot the genuine time embeddings of the Dev. set ===

    plotting_genuine(time_emb["time_embed_dev_05_samples_2D"], time_emb["chosen_labels_2_1_sex_asvspoof5"], time_emb["chosen_labels_2_1_attack_logical_mapping_asvspoof5"], \
        plot_title='UMAP - Train on Trn. ASVSpoof2019 \n Time Embeddings Dev. ASVSpoof5', gender='both', frontsize=20)
    
    # === plot the genuine time embeddings of the Eval. set ===

    plotting_genuine(time_emb["time_embed_eval_05_samples_2D"], time_emb["chosen_labels_3_1_sex_asvspoof5"], time_emb["chosen_labels_3_1_attack_logical_mapping_asvspoof5"], \
        plot_title='UMAP - Train on Trn. ASVSpoof2019 \n Time Embeddings Eval. ASVSpoof5', gender='both', frontsize=20)

    # === plot the genuine time embeddings of the Eval. set by without the codecs ===

    plotting_genuine_by_without_codec(time_emb["time_embed_eval_05_samples_2D"],time_emb["chosen_labels_3_1_sex_asvspoof5"],\
        time_emb["chosen_labels_3_1_attack_logical_mapping_asvspoof5"], \
        asvspoof5_eval_protocol['CODEC'], asvspoof5_eval_protocol['CODEC'].unique(),\
        plot_title='UMAP - Train on ASVSpoof2019 \n Time Embeddings Eval. ASVSpoof5', gender='both', frontsize=20)
    
    


