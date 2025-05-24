import numpy as np
import pandas as pd
import scipy.io as sio
from Histogram_calculations.similarity_measures_class import similarity_measures_class  
import os
# Constants
num_bins = 2**16
bin_edges = np.linspace(-1, 1, num_bins)



def compute_pmf_similarity_measures(pmfs: dict, similarity_measures_types : dict,output_filename: str = None ):
    """
    Compute similarity measures metrics for PMF pairs using specified similarity measures functions.

    Args:
        pmfs (dict): A dictionary with keys as labels and values as (pmf1, pmf2) tuples.
        similarity_measures_types (dict): A class containing static similarity measures methods.
        output_filename (str, optional): Path to save the result as an Excel file.

    Returns:
        pd.DataFrame: A DataFrame with computed similarity measures.
    """
    
   

    # Compute similarity measures
    results = []
    labels = list(pmfs.keys())
    for name, func in similarity_measures_types.items():
        row = [name]
        for key in labels:
            p1, p2 = pmfs[key]
            row.append(func(p1, p2))
        results.append(row)

    # Build column names
    columns = ['DistanceType'] + labels

    # Create DataFrame
    df = pd.DataFrame(results, columns=columns)

    # Save to Excel if requested
    if output_filename:
        df.to_csv(output_filename, index=False)
        print(f"Table saved to {output_filename}")

    return df


if __name__ == "__main__":

    similarity_measures_types = {
        'Symmetric KL': similarity_measures_class.kl_dist,
         'Modified Kolmogorov-Smirnov': similarity_measures_class.ks2_variant,
        'Hellinger': similarity_measures_class.hellinger,
    }
    
    # Set working directory
    os.chdir('./Histogram_calculations')
    TRAIN_FOLDER_DATA_2019 = './ASVspoof2019_train/'
    DEV_FOLDER_DATA_2019 = './ASVspoof2019_Dev/'
    Eval_FOLDER_DATA_2019 = './ASVspoof2019_Eval/'
    TRAIN_FOLDER_DATA_5 = './ASVspoof5_train/'
    DEV_FOLDER_DATA_5 = './ASVspoof5_Dev/'
    Eval_FOLDER_DATA_5 = './ASVspoof5_Eval/'
    
    # Load PMF data from .npy files
    pmfs = {
        'train_2019': (
            np.load(os.path.join(TRAIN_FOLDER_DATA_2019, 'train_data_ASVspoof2019_pmf_probs_bonafide.npy')),   # Bonafide PMF for ASVspoof2019 training set
            np.load(os.path.join(TRAIN_FOLDER_DATA_2019, 'train_data_ASVspoof2019_pmf_probs_spoofed.npy'))     # Spoofed PMF for ASVspoof2019 training set
        ),
        'train_5': (
            np.load(os.path.join(TRAIN_FOLDER_DATA_5, 'train_data_ASVspoof5_pmf_probs_bonafide.npy')),   # Bonafide PMF for ASVspoof5 training set
            np.load(os.path.join(TRAIN_FOLDER_DATA_5, 'train_data_ASVspoof5_pmf_probs_spoofed.npy'))     # Spoofed PMF for ASVspoof5 training set
        ),
        'validation_2019': (
            np.load(os.path.join(DEV_FOLDER_DATA_2019, 'validation_data_ASVspoof2019_pmf_probs_bonafide.npy')),  # Bonafide PMF for ASVspoof2019 dev set
            np.load(os.path.join(DEV_FOLDER_DATA_2019, 'validation_data_ASVspoof2019_pmf_probs_spoofed.npy'))    # Spoofed PMF for ASVspoof2019 dev set
        ),
         'validation_5': (
            np.load(os.path.join(DEV_FOLDER_DATA_5, 'validation_data_ASVspoof5_pmf_probs_bonafide.npy')),  # Bonafide PMF for ASVspoof5 dev set
            np.load(os.path.join(DEV_FOLDER_DATA_5, 'validation_data_ASVspoof5_pmf_probs_spoofed.npy'))    # Spoofed PMF for ASVspoof5 dev set
        ),
        'eval_2019': (
            np.load(os.path.join(Eval_FOLDER_DATA_2019, 'eval_data_ASVspoof2019_pmf_probs_bonafide.npy')),     # Bonafide PMF for ASVspoof2019 eval set
            np.load(os.path.join(Eval_FOLDER_DATA_2019, 'eval_data_ASVspoof2019_pmf_probs_spoofed.npy'))       # Spoofed PMF for ASVspoof2019 eval set
        ),
        'eval_5': (
            np.load(os.path.join(Eval_FOLDER_DATA_5, 'eval_data_ASVspoof5_pmf_probs_bonafide.npy')),    # Bonafide PMF for ASVspoof5 eval set
            np.load(os.path.join(Eval_FOLDER_DATA_5, 'eval_data_ASVspoof5_pmf_probs_spoofed.npy'))      # Spoofed PMF for ASVspoof5 eval set
        ),
    }

    # Compute similarity measures and save to Excel
    df = compute_pmf_similarity_measures(pmfs, similarity_measures_types, output_filename='Table2.csv')
    
    
    # Load PMF data from .npy files
    pmfs = {
        
        'bonafide_train_5': (
            np.load(os.path.join(TRAIN_FOLDER_DATA_2019, 'train_data_ASVspoof2019_pmf_probs_bonafide.npy')), # Bonafide PMF for ASVspoof2019 training set
            np.load(os.path.join(TRAIN_FOLDER_DATA_5, 'train_data_ASVspoof5_pmf_probs_bonafide.npy')),   # Bonafide PMF for ASVspoof5 training set
        ),
        'bonafide_validation_2019': (
            np.load(os.path.join(TRAIN_FOLDER_DATA_2019, 'train_data_ASVspoof2019_pmf_probs_bonafide.npy')), # Bonafide PMF for ASVspoof2019 training set
            np.load(os.path.join(DEV_FOLDER_DATA_2019, 'validation_data_ASVspoof2019_pmf_probs_bonafide.npy'))  # Bonafide PMF for ASVspoof2019 dev set
        ),
         'bonafide_validation_5': (
            np.load(os.path.join(TRAIN_FOLDER_DATA_2019, 'train_data_ASVspoof2019_pmf_probs_bonafide.npy')), # Bonafide PMF for ASVspoof2019 training set
            np.load(os.path.join(DEV_FOLDER_DATA_5, 'validation_data_ASVspoof5_pmf_probs_bonafide.npy'))  # Bonafide PMF for ASVspoof5 dev set
        ),
        'bonafide_eval_2019': (
            np.load(os.path.join(TRAIN_FOLDER_DATA_2019, 'train_data_ASVspoof2019_pmf_probs_bonafide.npy')), # Bonafide PMF for ASVspoof2019 training set
            np.load(os.path.join(Eval_FOLDER_DATA_2019, 'eval_data_ASVspoof2019_pmf_probs_bonafide.npy'))     # Bonafide PMF for ASVspoof2019 eval set
        ),
        'bonafide_eval_5': (
            np.load(os.path.join(TRAIN_FOLDER_DATA_2019, 'train_data_ASVspoof2019_pmf_probs_bonafide.npy')), # Bonafide PMF for ASVspoof2019 training set
            np.load(os.path.join(Eval_FOLDER_DATA_5, 'eval_data_ASVspoof5_pmf_probs_bonafide.npy'))   # Bonafide PMF for ASVspoof5 eval set
        ),
    }

    # Compute similarity measures and save to Excel
    df = compute_pmf_similarity_measures(pmfs, similarity_measures_types, output_filename='Table3.csv')

